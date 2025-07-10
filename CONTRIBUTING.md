# 🧠 Contributing to Krishna's Personal AI Assistant

Thanks for your interest in contributing! This project is a **Retrieval-Augmented Generation (RAG)**-based personal assistant powered by hybrid search (FAISS + BM25), multi-model LLM orchestration (OpenAI + NVIDIA), and memory-augmented context using LangChain. Contributions are welcome across **bug fixes**, **feature additions**, **prompt tuning**, and **infrastructure improvements**.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/personal-rag-assistant.git
cd personal-rag-assistant
```

### 2. Install Dependencies

Use Python 3.10+ and run:

```bash
pip install -r requirements.txt
```

Also ensure you have:
- `.env` file with valid `NVIDIA_API_KEY` and `OPENAI_API_KEY`
- `faiss_store/` directory with a valid FAISS index
- `all_chunks.json` with the retrieved document data

---

## 🧩 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Main Gradio app, chains, models, prompts, memory handling |
| `agentic_runner.py` | CLI agent wrapper around main tools |
| `Vector_storing.py` | Vector generation and FAISS index builder |
| `all_chunks.json` | Stores document chunks metadata and content |
| `faiss_store/` | Contains FAISS vector index files |

---

## 💡 How to Contribute

### 🔧 Bug Fixes or Enhancements
1. Open an issue describing the problem or idea.
2. Fork the repo and create a new branch.
3. Submit a pull request (PR) with clear explanation and test results.

### ✨ Add New Features
- Add new chains, tools, or prompt types in `app.py`.
- Ensure they are modular and reusable.
- Use the `RunnableLambda`, `RunnableAssign`, and LangChain patterns already established.

### 🧪 Testing Your Work
- Use `main.ipynb` for quick experiments.
- Run `python agentic_runner.py` to simulate CLI interactions.
- Launch the Gradio UI for full testing:

```bash
python app.py
```

---

## 📦 Code Style & Guidelines

- Follow PEP8 formatting.
- Use type hints and Pydantic models where applicable.
- Prompt templates should be clean, instructive, and tested.
- Prefer `ChatPromptTemplate.from_template` for reusability.

---

## 🔐 Secrets & Keys

**DO NOT hardcode secrets.** Store them in `.env`:

```env
NVIDIA_API_KEY=your_key
OPENAI_API_KEY=your_key
```

---

## 🙋‍♂️ Questions?

Open an issue or reach out to [Krishna Vamsi Dhulipalla](mailto:krishnavamsi@vt.edu)

---

Let’s build the future of personal LLM assistants together 🚀
