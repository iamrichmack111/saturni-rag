#!/usr/bin/env python3
import os, glob, argparse, faiss, numpy as np, pickle, torch
from transformers import AutoTokenizer, AutoModel
from pyfiglet import Figlet
from termcolor import colored

# --- Banner ---
print(colored(Figlet(font="slant").renderText("Saturni"), "cyan"))

INDEX_FILE, META_FILE, MODEL_FILE = "books.faiss", "meta.pkl", "model.pkl"

# --- Model Loader ---
def load_model(name):
    print(colored(f"🔮 Loading model: {name}", "yellow"))
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tok, model

# Load saved model choice if exists
if os.path.exists(MODEL_FILE):
    model_name = pickle.load(open(MODEL_FILE, "rb"))
else:
    model_name = "google/gemma-2b"

tok, model = load_model(model_name)

def embed(text, maxlen=500):
    with torch.no_grad():
        t = tok(text, return_tensors="pt", truncation=True, max_length=maxlen)
        return model(**t).last_hidden_state.mean(1).cpu().numpy()

def chunk_text(text, size=500):
    toks = tok(text, return_tensors="pt")["input_ids"][0]
    return [tok.decode(toks[i:i+size]) for i in range(0, len(toks), size)]

def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise RuntimeError("❌ No index found. Run with --index first.")
    return faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE,"rb"))

def build_index():
    files = glob.glob("pg*.txt")
    if not files: return print(colored("❌ No .txt files found", "red"))
    docs, vecs = [], []
    for f in files:
        with open(f, encoding="utf8", errors="ignore") as fh:
            txt = fh.read()
        for ch in chunk_text(txt):
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
        for ch in chunk_text(txt):
            new_docs.append((os.path.basename(f), ch))
            new_vecs.append(embed(ch))
    new_vecs = np.vstack(new_vecs).astype("float32")
    if index is None: index = faiss.IndexFlatL2(new_vecs.shape[1])
    index.add(new_vecs); docs.extend(new_docs)
    faiss.write_index(index, INDEX_FILE); pickle.dump(docs, open(META_FILE,"wb"))
    print(colored(f"✅ Added {len(new_docs)} chunks from {len(files)} files", "yellow"))

def query(q, topk=3):
    index, docs = load_index()
    qv = embed(q).astype("float32")
    D,I = index.search(qv, topk)
    return [docs[idx] for idx in I[0]]

def repl():
    print(colored("💬 Enter questions (type 'exit' to quit)\n", "magenta"))
    while True:
        q = input(colored("❓ Ask> ", "blue"))
        if q.lower() in {"exit","quit"}: break
        for (fname, ans) in query(q):
            print(colored("📖 "+fname, "yellow"))
            print(colored("👉 "+ans[:200]+" ...", "green"))

if __name__=="__main__":
    ap = argparse.ArgumentParser(
        description="📚 Saturni: Flexible RAG CLI with FAISS + Transformers",
        epilog="Examples:\n"
               "  ./saturni.py --index\n"
               "  ./saturni.py --add new.txt\n"
               "  ./saturni.py --repl\n"
               "  ./saturni.py --query 'What is Faust about?'\n"
               "  ./saturni.py --model sentence-transformers/all-MiniLM-L6-v2",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--index", action="store_true", help="Build FAISS index from all pg*.txt files")
    ap.add_argument("--add", nargs="+", help="Add new .txt files into existing FAISS index")
    ap.add_argument("--repl", action="store_true", help="Interactive query mode")
    ap.add_argument("--query", type=str, help="One-shot query without REPL")
    ap.add_argument("--model", type=str, help="Change embedding model (default: google/gemma-2b)")
    args = ap.parse_args()

    # Change model
    if args.model:
        model_name = args.model
        tok, model = load_model(model_name)
        pickle.dump(model_name, open(MODEL_FILE,"wb"))
        print(colored(f"✅ Model saved: {model_name}", "cyan"))

    if args.index: build_index()
    if args.add: add_files(args.add)
    if args.repl: repl()
    if args.query:
        for (fname, ans) in query(args.query):
            print(colored("📖 "+fname, "yellow"))
            print(colored("👉 "+ans[:200]+" ...", "green"))

