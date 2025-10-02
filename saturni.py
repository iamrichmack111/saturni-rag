#!/usr/bin/env python3
import os, glob, argparse, faiss, numpy as np, pickle, subprocess, requests
from pyfiglet import Figlet
from termcolor import colored

# --- Banner ---
print(colored(Figlet(font="slant").renderText("Saturni"), "cyan"))

INDEX_FILE, META_FILE = "books.faiss", "meta.pkl"

# --- Ensure Ollama Model ---
def ensure_ollama_model(model_name):
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in out.stdout:
            print(colored(f"📥 Pulling Ollama model: {model_name}", "yellow"))
            subprocess.run(["ollama", "pull", model_name], check=True)
    except Exception as e:
        print(colored(f"❌ Could not verify/pull Ollama model: {e}", "red"))

# --- Embedding with Ollama ---
def embed(text, model="gemma2:2b"):
    ensure_ollama_model(model)
    try:
        resp = requests.post("http://localhost:11434/api/embed",
            json={"model": model, "input": text})
        return np.array(resp.json()["embedding"], dtype="float32").reshape(1,-1)
    except Exception as e:
        raise RuntimeError(f"❌ Ollama embedding failed: {e}")

# --- FAISS Index ---
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise RuntimeError("❌ No index found. Run with --index first.")
    return faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE,"rb"))

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index(model="gemma2:2b"):
    files = glob.glob("pg*.txt")
    if not files: return print(colored("❌ No .txt files found", "red"))
    docs, vecs = [], []
    for f in files:
        with open(f, encoding="utf8", errors="ignore") as fh: txt = fh.read()
        for ch in chunk_text(txt):
            docs.append((os.path.basename(f), ch))
            vecs.append(embed(ch, model))
    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)
    faiss.write_index(index, INDEX_FILE); pickle.dump(docs, open(META_FILE,"wb"))
    print(colored(f"✅ Indexed {len(docs)} chunks from {len(files)} files", "green"))

def add_files(files, model="gemma2:2b"):
    index, docs = (faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE,"rb"))) if os.path.exists(INDEX_FILE) else (None, [])
    new_docs, new_vecs = [], []
    for f in files:
        with open(f, encoding="utf8", errors="ignore") as fh: txt = fh.read()
        for ch in chunk_text(txt):
            new_docs.append((os.path.basename(f), ch))
            new_vecs.append(embed(ch, model))
    new_vecs = np.vstack(new_vecs).astype("float32")
    if index is None: index = faiss.IndexFlatL2(new_vecs.shape[1])
    index.add(new_vecs); docs.extend(new_docs)
    faiss.write_index(index, INDEX_FILE); pickle.dump(docs, open(META_FILE,"wb"))
    print(colored(f"✅ Added {len(new_docs)} chunks from {len(files)} files", "yellow"))

def retrieve(q, topk=3, model="gemma2:2b"):
    index, docs = load_index()
    qv = embed(q, model).astype("float32")
    D,I = index.search(qv, topk)
    return [docs[idx] for idx in I[0]]

# --- Ollama LLM Answer ---
def ollama_generate(prompt, model="gemma2:2b"):
    ensure_ollama_model(model)
    try:
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt}, stream=True)
        out = ""
        for line in resp.iter_lines():
            if line:
                part = line.decode("utf-8")
                if '"response":"' in part:
                    out += part.split('"response":"')[1].split('"')[0]
        return out.strip()
    except Exception as e:
        return f"❌ Ollama request failed: {e}"

def rag_answer(q, model="gemma2:2b"):
    ctx = retrieve(q, topk=3, model=model)
    context_text = "\n\n".join([f"From {fname}:\n{chunk}" for fname, chunk in ctx])
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context_text}\n\nQuestion: {q}\nAnswer:"
    return ollama_generate(prompt, model=model)

# --- CLI Modes ---
def repl(ai_model):
    print(colored(f"💬 Ask questions (AI model = {ai_model}, type 'exit' to quit)\n", "magenta"))
    while True:
        q = input(colored("❓ Ask> ", "blue"))
        if q.lower() in {"exit","quit"}: break
        ans = rag_answer(q, model=ai_model)
        print(colored("👉 "+ans, "green"))

if __name__=="__main__":
    ap = argparse.ArgumentParser(
        description="📚 Saturni: FAISS RAG + Ollama AI",
        epilog="Examples:\n"
               "  ./saturni.py --index\n"
               "  ./saturni.py --add new.txt\n"
               "  ./saturni.py --repl --ai gemma2:2b\n"
               "  ./saturni.py --query 'What is Faust about?' --ai mistral:7b",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--index", action="store_true", help="Build FAISS index from all pg*.txt files")
    ap.add_argument("--add", nargs="+", help="Add new .txt files into existing FAISS index")
    ap.add_argument("--repl", action="store_true", help="Interactive query mode with Ollama AI")
    ap.add_argument("--query", type=str, help="One-shot query with Ollama AI")
    ap.add_argument("--ai", type=str, default="gemma2:2b", help="Ollama model to use (default: gemma2:2b)")
    args = ap.parse_args()

    if args.index: build_index(args.ai)
    if args.add: add_files(args.add, args.ai)
    if args.repl: repl(args.ai)
    if args.query:
        ans = rag_answer(args.query, model=args.ai)
        print(colored("👉 "+ans, "green"))

