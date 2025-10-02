#!/usr/bin/env python3
import os, glob, argparse, faiss, numpy as np, pickle, subprocess, requests, json
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm   # progress bar

print(colored(Figlet(font="slant").renderText("Saturni"), "cyan"))

INDEX_FILE, META_FILE = "books.faiss", "meta.pkl"
EMBED_MODEL = "nomic-embed-text"   # embedding model

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

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

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
    faiss.write_index(index, INDEX_FILE); pickle.dump(docs, open(META_FILE,"wb"))
    print(colored(f"✅ Indexed {len(docs)} chunks from {len(files)} files", "green"))

def add_files(files):
    index, docs = (faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE,"rb"))) if os.path.exists(INDEX_FILE) else (None, [])
    new_docs, new_vecs = [], []
    for f in files:
        with open(f, encoding="utf8", errors="ignore") as fh: txt = fh.read()
        chunks = chunk_text(txt)
        for ch in tqdm(chunks, desc=f"Embedding {f}", unit="chunk"):
            new_docs.append((os.path.basename(f), ch))
            new_vecs.append(embed(ch))
    new_vecs = np.vstack(new_vecs).astype("float32")
    if index is None: index = faiss.IndexFlatL2(new_vecs.shape[1])
    index.add(new_vecs); docs.extend(new_docs)
    faiss.write_index(index, INDEX_FILE); pickle.dump(docs, open(META_FILE,"wb"))
    print(colored(f"✅ Added {len(new_docs)} chunks from {len(files)} files", "yellow"))

def retrieve(q, topk=3):
    index, docs = load_index()
    qv = embed(q).astype("float32")
    D,I = index.search(qv, topk)
    return [docs[idx] for idx in I[0]]

# --- Ollama LLM Answer ---
def ollama_generate(prompt, model="gemma2:2b"):
    ensure_ollama_model(model)
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}, stream=True)
    out = ""
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    out += data["response"]
            except:
                pass
    return out.strip()

def rag_answer(q, model="gemma2:2b"):
    ctx = retrieve(q, topk=3)
    context_text = "\n\n".join([f"From {fname}:\n{chunk}" for fname, chunk in ctx])
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context_text}\n\nQuestion: {q}\nAnswer:"
    return ollama_generate(prompt, model=model)

# --- CLI Modes ---
def output_result(text, output_file=None):
    print(colored("👉 " + text, "green"))
    if output_file:
        with open(output_file, "w", encoding="utf8") as f:
            f.write(text)
        print(colored(f"💾 Saved output to {output_file}", "cyan"))

def repl(ai_model, output_file=None):
    print(colored(f"💬 Ask questions (AI model = {ai_model}, type 'exit' to quit)\n","magenta"))
    while True:
        q = input(colored("❓ Ask> ", "blue"))
        if q.lower() in {"exit","quit"}: break
        ans = rag_answer(q, model=ai_model)
        output_result(ans, output_file)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="📚 Saturni: FAISS RAG + Ollama AI with progress bar and output")
    ap.add_argument("--index", action="store_true", help="Build FAISS index")
    ap.add_argument("--add", nargs="+", help="Add new .txt files")
    ap.add_argument("--repl", action="store_true", help="Interactive query mode")
    ap.add_argument("--query", type=str, help="One-shot query")
    ap.add_argument("--ai", type=str, default="gemma2:2b", help="Ollama model for answers (default gemma2:2b)")
    ap.add_argument("-o", "--output", type=str, help="Save output to file")
    args = ap.parse_args()

    if args.index: build_index()
    if args.add: add_files(args.add)
    if args.repl: repl(args.ai, args.output)
    if args.query:
        ans = rag_answer(args.query, model=args.ai)
        output_result(ans, args.output)

