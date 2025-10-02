Here’s a **README.md** draft and dependency list tailored for your Saturni RAG project.
It includes install/setup steps, dependencies, usage examples, author credit, and the GitHub repo info you gave me.

---

## 📦 Dependencies (add to `requirements.txt`)

```txt
faiss-cpu
numpy
requests
pyfiglet
termcolor
tqdm
```

If you want to lock versions:

```txt
faiss-cpu==1.8.0
numpy>=1.26
requests>=2.31
pyfiglet>=1.0.2
termcolor>=2.4
tqdm>=4.66
```

---

## 📜 README.md

````markdown
# Saturni RAG

**Saturni** is a lightweight Retrieval-Augmented Generation (RAG) system designed for analyzing philosophy (and other texts).  
It uses **Ollama** for embeddings and generation, **FAISS** for fast similarity search, and a colorful **CLI** with logging.

---

## ✨ Features

- 🚀 Build a FAISS index from local text files (`clean_pg*.txt` or `pg*.txt`)
- 🔎 Retrieve relevant chunks and feed them into an Ollama model
- 💬 Interactive REPL or one-shot query mode
- 📥 Automatic `ollama pull` if a model isn’t installed
- 📊 Progress bars for indexing and generation
- 📝 Transcript logging with `-o` (append Q&A to a log file)
- 🎨 Figlet banner + multicolored CLI output

---

## 📦 Dependencies

Install with pip:

```bash
pip install -r requirements.txt
````

Dependencies:

* `faiss-cpu` – vector search
* `numpy` – numerical arrays
* `requests` – HTTP calls to Ollama
* `pyfiglet` – ASCII art banner
* `termcolor` – colored terminal output
* `tqdm` – progress bars

---

## ⚙️ Setup

1. Install [Ollama](https://ollama.ai) and run it:

   ```bash
   ollama serve
   ```

2. Clone this repo:

   ```bash
   git clone git@github.com:iamrichmack111/saturni-rag.git
   cd saturni-rag
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place your philosophy texts into the repo directory.
   They should be named either `pg1234.txt` or `clean_pg1234.txt`.

---

## 🔑 Usage

### Build the index

```bash
./saturni.py --index
```

### Add new texts

```bash
./saturni.py --add clean_pg9999.txt
```

### Query interactively (with logging)

```bash
./saturni.py --repl --ai gemma2:2b -o session.log
```

### One-shot query

```bash
./saturni.py --query "Summarize Nietzsche’s Beyond Good and Evil" --ai mistral:7b -o output.log
```

Output is both printed to screen and appended to the log.

---

## 📂 File structure

```
saturni-rag/
│── saturni.py         # main CLI
│── books.faiss        # FAISS index (generated after --index)
│── meta.pkl           # chunk metadata
│── requirements.txt   # dependencies
│── README.md          # this file
│── clean_pg*.txt      # your philosophy texts
```

---

## 👤 Author

* **Jeremy Franklin**
* GitHub: [iamrichmack111](https://github.com/iamrichmack111)
* Repo: [git@github.com:iamrichmack111/saturni-rag.git](git@github.com:iamrichmack111/saturni-rag.git)

---

## 📜 License

MIT License — feel free to modify and share.

```

---


```

