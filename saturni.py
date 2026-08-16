#!/usr/bin/env python3
import os, glob, argparse, faiss, numpy as np, pickle, subprocess, requests, json
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm   # progress bar

print(colored(Figlet(font="slant").renderText("Saturni"), "cyan"))

INDEX_FILE, META_FILE, CONFIG_FILE = "books.faiss", "meta.pkl", "index_config.pkl"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 200

# --- Ensure Ollama Model ---
def ensure_ollama_model(model_name):
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name not in out.stdout:
        print(colored(f"📥 Pulling Ollama model: {model_name}", "yellow"))
        subprocess.run(["ollama", "pull", model_name], check=True)

# --- Embedding with Ollama ---
def embed(text):
    ensure_ollama_model(EMBED_MODEL)
    resp = requests.post("http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text}).json()
    return np.array(resp["embeddings"][0], dtype="float32").reshape(1,-1)

# --- Helpers ---
def find_text_files():
    return glob.glob("clean_pg*.txt") or glob.glob("pg*.txt")

def chunk_text(text, size=None):
    size = CHUNK_SIZE if size is None else size
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def current_index_config():
    return {
        "chunk_size": CHUNK_SIZE,
        "embed_model": EMBED_MODEL,
    }

def save_index_config():
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(current_index_config(), f)

def validate_index_config():
    if not os.path.exists(CONFIG_FILE):
        raise RuntimeError(
            "Existing index has no configuration metadata. "
            "Create index_config.pkl for the existing index or rebuild with --index."
        )

    with open(CONFIG_FILE, "rb") as f:
        stored = pickle.load(f)

    requested = current_index_config()

    if stored != requested:
        raise RuntimeError(
            "Index configuration mismatch. "
            f"Stored={stored}, requested={requested}. "
            "Rebuild the index before changing chunk or embedding settings."
        )

# --- FAISS Index ---
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise RuntimeError("❌ No index found. Run with --index first.")
    return faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE,"rb"))

def build_index():
    files = find_text_files()
    if not files: return print(colored("❌ No .txt files found", "red"))
    docs, vecs = [], []
    for f in files:
        with open(f, encoding="utf8", errors="ignore") as fh: txt = fh.read()
        chunks = chunk_text(txt)
        for ch in tqdm(chunks, desc=f"Embedding {f}", unit="chunk"):
            docs.append((os.path.basename(f), ch))
            vecs.append(embed(ch))
    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)
    faiss.write_index(index, INDEX_FILE)
    pickle.dump(docs, open(META_FILE, "wb"))
    save_index_config()
    print(colored(f"✅ Indexed {len(docs)} chunks from {len(files)} files", "green"))

def add_files(files):
    if os.path.exists(INDEX_FILE):
        validate_index_config()

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        docs = pickle.load(open(META_FILE, "rb"))
    else:
        docs = []

    files = [os.path.abspath(f) for f in files]
    names = {os.path.basename(f) for f in files}

    retained_docs = [(fname, ch) for fname, ch in docs if fname not in names]
    replaced = len(docs) - len(retained_docs)

    new_docs = []
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"File not found: {f}")

        with open(f, encoding="utf8", errors="ignore") as fh:
            txt = fh.read()

        chunks = chunk_text(txt)
        for ch in tqdm(chunks, desc=f"Embedding {f}", unit="chunk"):
            new_docs.append((os.path.basename(f), ch))

    all_docs = retained_docs + new_docs

    if not all_docs:
        raise RuntimeError("No chunks available to index.")

    vecs = []
    for _, ch in tqdm(all_docs, desc="Rebuilding FAISS", unit="chunk"):
        vecs.append(embed(ch))

    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, INDEX_FILE)
    pickle.dump(all_docs, open(META_FILE, "wb"))

    if replaced:
        print(colored(
            f"✅ Updated {len(files)} file(s): replaced {replaced} old chunks "
            f"with {len(new_docs)} new chunks",
            "yellow"
        ))
    else:
        print(colored(
            f"✅ Added {len(new_docs)} chunks from {len(files)} file(s)",
            "yellow"
        ))

def retrieve(q, topk=3, max_distance=0.90):
    index, docs = load_index()
    qv = embed(q).astype("float32")
    D, I = index.search(qv, topk)

    results = []
    for distance, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if float(distance) <= max_distance:
            fname, chunk = docs[idx]
            results.append((fname, chunk, float(distance)))

    return results

# --- Ollama LLM Answer (with progress bar) ---
def ollama_generate(prompt, model="gemma2:2b"):
    ensure_ollama_model(model)
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}, stream=True)
    out = ""
    with tqdm(desc="Generating answer", unit="chunk") as pbar:
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        out += data["response"]
                        pbar.update(1)
                except:
                    pass
    return out.strip()

def rag_answer(q, model="gemma2:2b", max_distance=0.90):
    ctx = retrieve(q, topk=3, max_distance=max_distance)

    if not ctx:
        return "Not found in text."

    context_text = "\n\n".join(
        f"From {fname} [distance={distance:.4f}]:\n{chunk}"
        for fname, chunk, distance in ctx
    )

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not support the answer, say: Not found in text.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"
    )

    return ollama_generate(prompt, model=model)

# --- CLI Modes ---
def output_result(question, answer, output_file=None):
    print(colored("👉 " + answer, "green"))
    if output_file:
        with open(output_file, "a", encoding="utf8") as f:  # append mode
            f.write(f"❓ {question}\n")
            f.write(f"👉 {answer}\n\n")
        print(colored(f"💾 Appended Q&A to {output_file}", "cyan"))

def repl(ai_model, output_file=None, max_distance=0.90):
    print(colored(f"💬 Ask questions (AI model = {ai_model}, type 'exit' to quit)\n","magenta"))
    while True:
        q = input(colored("❓ Ask> ", "blue"))
        if q.lower() in {"exit","quit"}: break
        ans = rag_answer(q, model=ai_model, max_distance=max_distance)
        output_result(q, ans, output_file)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="📚 Saturni: FAISS RAG + Ollama AI with transcript logging")
    ap.add_argument("--index", action="store_true", help="Build FAISS index")
    ap.add_argument("--add", nargs="+", help="Add new .txt files")
    ap.add_argument("--repl", action="store_true", help="Interactive query mode")
    ap.add_argument("--query", type=str, help="One-shot query")
    ap.add_argument("--ai", type=str, default="gemma2:2b", help="Ollama model for answers (default gemma2:2b)")
    ap.add_argument("--max-distance", type=float, default=0.90,
                    help="Maximum FAISS L2 retrieval distance (lower is better; default 0.90)")
    ap.add_argument("-o", "--output", type=str, help="Append Q&A transcript to file")
    args = ap.parse_args()

    if args.index: build_index()
    if args.add: add_files(args.add)
    if args.repl: repl(args.ai, args.output, args.max_distance)
    if args.query:
        ans = rag_answer(args.query, model=args.ai, max_distance=args.max_distance)
        output_result(args.query, ans, args.output)

