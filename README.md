# 🧠 Krishna's Personal AI Chatbot

A memory-grounded, retrieval-augmented AI assistant built with LangChain, FAISS, BM25, and Llama3 — personalized to Krishna Vamsi Dhulipalla’s career, projects, and technical profile.

> ⚡️ Ask me anything about Krishna — skills, experience, goals, or even what tools he used at Virginia Tech.

---

## 📌 Features

- ✅ **Hybrid Retrieval**: Combines dense vector search (FAISS) + keyword search (BM25) for precise, high-recall chunk selection
- 🤖 **LLM-Powered Pipelines**: Uses OpenAI GPT-4o and NVIDIA NIMs (e.g. LLaMA-3, Mixtral) for rewriting, validation, and final answer generation
- 🧠 **Memory Module**: Stores user preferences, recent topics, and inferred tone using a structured `KnowledgeBase` schema
- 🛠️ **Custom Architecture**:
  - Query → Rewriting → Hybrid Retriever → Scope Validator → LLM Answer
  - Fallback humor model (Mixtral) for out-of-scope queries
- 🧩 **Document Grounding**: Powered by Krishna’s actual markdown files like `profile.md`, `goals.md`, and `chatbot_architecture.md`
- 📊 **Enriched Vector Store**: Chunks include LLM-generated summaries and synthetic queries for better search performance
- 🎛️ **Gradio Frontend**: Responsive, markdown-formatted interface for natural, real-time interaction

---

## 🏗️ Architecture

```text
User Query
   ↓
[LLM1] → Rephrase into 3 diverse subqueries
   ↓
Hybrid Retrieval (BM25 + FAISS)
   ↓
[LLM2] → Classify: In-scope or Out-of-scope
   ↓
   ├─ In-scope → Top-k Chunks → GPT-4o
   └─ Out-of-scope → Mixtral (funny fallback)
   ↓
Final Answer + Async Memory Update
```

---

## 📂 Project Structure

```
.
├── app.py                      # Main Gradio app and pipeline logic
├── Vector_storing.py          # Chunking, LLM-based enrichment, and FAISS store creation
├── requirements.txt           # Python package dependencies
├── faiss_store/               # Saved FAISS vector index
├── all_chunks.json            # JSON of enriched document chunks
├── personal_data/             # Source markdown files (right now excluded)
├── README.md
```

---

## 🧠 Knowledge Sources

All answers are grounded in curated markdown files:

| File Name                 | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `profile.md`              | Krishna’s full technical profile and education |
| `goals.md`                | Short- and long-term personal goals            |
| `chatbot_architecture.md` | System-level breakdown of this AI assistant    |
| `personal_interests.md`   | Hobbies, cultural identity, food preferences   |
| `conversations.md`        | Sample queries and expected response tone      |

---

## 🧪 How It Works

1. **User input** is rewritten into subqueries (LLM1)
2. **Retriever** fetches relevant chunks using BM25 and FAISS
3. **Classifier LLM** decides if results are relevant to Krishna
4. **GPT-4o** generates final answer using top-k chunks
5. **Memory is updated** asynchronously with every turn

---

## 💬 Example Queries

- What programming languages does Krishna know?
- Tell me about Krishna’s chatbot architecture
- Can this chatbot explain Krishna's work at Virginia Tech?
- What tools has Krishna used for data engineering?

---

## 🚀 Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/krishna-creator/krishna-personal-chatbot.git
cd krishna-personal-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys (OpenAI, NVIDIA)
export OPENAI_API_KEY=...
export NVIDIA_API_KEY=...

# 4. Launch the chatbot
python app.py
```

---

## 🔮 Model Stack

| Purpose            | Model Name               | Provider |
| ------------------ | ------------------------ | -------- |
| Query Rewriting    | `phi-3-mini-4k-instruct` | NVIDIA   |
| Scope Classifier   | `llama-3-70b-instruct`   | NVIDIA   |
| Answer Generator   | `gpt-4o`                 | OpenAI   |
| Fallback Humor LLM | `mixtral-8x22b-instruct` | NVIDIA   |

---

## 📌 Acknowledgments

- Built as part of Krishna's exploration into **LLM orchestration and agentic RAG**
- Inspired by LangChain, SentenceTransformers, and NVIDIA RAG Agents Course

---

## 📜 License

MIT License © Krishna Vamsi Dhulipalla
