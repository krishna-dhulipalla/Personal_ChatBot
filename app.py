import os
import json
import re
import hashlib
from functools import partial
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.style import Style
from langchain_core.runnables import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)
api_key = os.getenv("NVIDIA_API_KEY")
os.environ["NVIDIA_API_KEY"] = api_key

# Constants
FAISS_PATH = "faiss_store/v30_600_150"
CHUNKS_PATH = "all_chunks.json"
KRISHNA_BIO = """Krishna Vamsi Dhulipalla is a graduate student in Computer Science at Virginia Tech (M.Eng, expected 2024), with over 3 years of experience across data engineering, machine learning research, and real-time analytics. He specializes in building scalable data systems and intelligent LLM-powered applications, with strong expertise in Python, PyTorch, Hugging Face Transformers, and end-to-end ML pipelines.

He has led projects involving retrieval-augmented generation (RAG), feature selection for genomic classification, fine-tuning domain-specific LLMs (e.g., DNABERT, HyenaDNA), and real-time forecasting systems using Kafka, Spark, and Airflow. His cloud proficiency spans AWS (S3, SageMaker, ECS, CloudWatch), GCP (BigQuery, Cloud Composer), and DevOps tools like Docker, Kubernetes, and MLflow.

Krishna’s academic focus areas include genomic sequence modeling, transformer optimization, MLOps automation, and cross-domain generalization. He has published research in bioinformatics and ML applications for circadian transcription prediction and transcription factor binding.

He is certified in NVIDIA’s RAG Agents with LLMs, Google Cloud Data Engineering, AWS ML Specialization, and has a proven ability to blend research and engineering in real-world systems. Krishna is passionate about scalable LLM infra, data-centric AI, and domain-adaptive ML solutions."""

def initialize_console():
    console = Console()
    base_style = Style(color="#76B900", bold=True)
    return partial(console.print, style=base_style)

pprint = initialize_console()

def load_chunks_from_json(path: str = CHUNKS_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_faiss(path: str = FAISS_PATH, 
               model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def initialize_resources():
    vectorstore = load_faiss()
    all_chunks = load_chunks_from_json()
    all_texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    return vectorstore, all_chunks, all_texts, metadatas

vectorstore, all_chunks, all_texts, metadatas = initialize_resources()

# LLMs
repharser_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3") | StrOutputParser()
relevance_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
answer_llm = ChatOpenAI(
    model="gpt-4-1106-preview",              
    temperature=0.3,             
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()] 
) | StrOutputParser()


# Prompts
repharser_prompt = ChatPromptTemplate.from_template(
    "Rewrite the question below in 4 diverse ways to retrieve semantically similar information.Ensure diversity in phrasings across style, voice, and abstraction:\n\nQuestion: {query}\n\nRewrites:"
)

relevance_prompt = ChatPromptTemplate.from_template("""
You are Krishna's personal AI assistant validator.
Your job is to review a user's question and a list of retrieved document chunks.
Identify which chunks (if any) directly help answer the question. Return **all relevant chunks**.

---
⚠️ Do NOT select chunks just because they include keywords or technical terms.

Exclude chunks that:
- Mention universities, CGPA, or education history (they show qualifications, not skills)
- List certifications or course names (they show credentials, not skills used)
- Describe goals, future plans, or job aspirations
- Contain tools mentioned in passing without describing actual usage

Only include chunks if they contain **evidence of specific knowledge, tools used, skills applied, or experience demonstrated.**

---

🔎 Examples:

Q1: "What are Krishna's skills?"
- Chunk A: Lists programming languages, ML tools, and projects → ✅
- Chunk B: Talks about a Coursera certificate in ML → ❌
- Chunk C: States a CGPA and master’s degree → ❌
- Chunk D: Describes tools Krishna used in his work → ✅

Output:
{{
  "valid_chunks": [A, D],
  "is_out_of_scope": false,
  "justification": "Chunks A and D describe tools and skills Krishna has actually used."
}}

Q2: "What is Krishna's favorite color?"
- All chunks are about technical work or academic history → ❌

Output:
{{
  "valid_chunks": [],
  "is_out_of_scope": true,
  "justification": "None of the chunks are related to the user's question about preferences or colors."
}}

---

Now your turn.

User Question:
"{query}"

Chunks:
{contents}

Return only the JSON object. Think carefully before selecting any chunk.
""")

answer_prompt_relevant = ChatPromptTemplate.from_template(
    "You are Krishna's personal AI assistant. Your job is to answer the user’s question clearly and professionally using the provided context.\n"
    "Rather than copying sentences, synthesize relevant insights and explain them like a knowledgeable peer.\n\n"
    "Krishna's Background:\n{profile}\n\n"
    "Make your response rich and informative by:\n"
    "- Combining relevant facts from multiple parts of the context\n"
    "- Using natural, human-style language (not just bullet points)\n"
    "- Expanding briefly on tools or skills when appropriate\n"
    "- Avoiding repetition, filler, or hallucinations\n\n"
    "Context:\n{context}\n\n"
    "User Question:\n{query}\n\n"
    "Answer:"
)

answer_prompt_fallback = ChatPromptTemplate.from_template(
    "You are Krishna’s personal AI assistant. The user asked a question unrelated to Krishna’s background.\n"
    "Gently let the user know, and then pivot to something Krishna is actually involved in to keep the conversation helpful.\n\n"
    "Krishna's Background:\n{profile}\n\n"
    "User Question:\n{query}\n\n"
    "Your Answer:"
)
# Helper Functions
def parse_rewrites(raw_response: str) -> list[str]:
    lines = raw_response.strip().split("\n")
    return [line.strip("0123456789. ").strip() for line in lines if line.strip()][:4]

def hybrid_retrieve(inputs, exclude_terms=None):
    # if exclude_terms is None:
    #     exclude_terms = ["cgpa", "university", "b.tech", "m.s.", "certification", "coursera", "edx", "goal", "aspiration", "linkedin", "publication", "ieee", "doi", "degree"]

    all_queries = inputs["all_queries"]
    bm25_retriever = BM25Retriever.from_texts(texts=all_texts, metadatas=metadatas)
    bm25_retriever.k = inputs["k_per_query"]
    vectorstore = inputs["vectorstore"]
    alpha = inputs["alpha"]
    top_k = inputs.get("top_k", 15)

    scored_chunks = defaultdict(lambda: {
        "vector_scores": [],
        "bm25_score": 0.0,
        "content": None,
        "metadata": None,
    })

    for subquery in all_queries:
        vec_hits = vectorstore.similarity_search_with_score(subquery, k=inputs["k_per_query"])
        for doc, score in vec_hits:
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            scored_chunks[key]["vector_scores"].append(score)
            scored_chunks[key]["content"] = doc.page_content
            scored_chunks[key]["metadata"] = doc.metadata

        bm_hits = bm25_retriever.invoke(subquery)
        for rank, doc in enumerate(bm_hits):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            bm_score = 1.0 - (rank / inputs["k_per_query"])
            scored_chunks[key]["bm25_score"] += bm_score
            scored_chunks[key]["content"] = doc.page_content
            scored_chunks[key]["metadata"] = doc.metadata

    all_vec_means = [np.mean(v["vector_scores"]) for v in scored_chunks.values() if v["vector_scores"]]
    max_vec = max(all_vec_means) if all_vec_means else 1
    min_vec = min(all_vec_means) if all_vec_means else 0

    final_results = []
    for chunk in scored_chunks.values():
        vec_score = np.mean(chunk["vector_scores"]) if chunk["vector_scores"] else 0.0
        norm_vec = (vec_score - min_vec) / (max_vec - min_vec) if max_vec != min_vec else 1.0
        bm25_score = chunk["bm25_score"] / len(all_queries)
        final_score = alpha * norm_vec + (1 - alpha) * bm25_score

        content = chunk["content"].lower()
        # if any(term in content for term in exclude_terms):
        #     continue
        if final_score < 0.05 or len(content.strip()) < 100:
            continue

        final_results.append({
            "content": chunk["content"],
            "source": chunk["metadata"].get("source", ""),
            "final_score": float(round(final_score, 4)),
            "vector_score": float(round(vec_score, 4)),
            "bm25_score": float(round(bm25_score, 4)),
            "metadata": chunk["metadata"],
            "summary": chunk["metadata"].get("summary", ""),
            "synthetic_queries": chunk["metadata"].get("synthetic_queries", [])
        })

    final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)

    seen = set()
    unique_chunks = []
    for chunk in final_results:
        clean_text = re.sub(r'\W+', '', chunk["content"].lower())[:300]
        fingerprint = (chunk["source"], clean_text)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_chunks.append(chunk)

    unique_chunks = unique_chunks[:top_k]

    return {
        "query": inputs["query"],
        "chunks": unique_chunks
    }
    
def safe_json_parse(s: str) -> Dict:
    return json.loads(s) if isinstance(s, str) and "valid_chunks" in s else {
        "valid_chunks": [], 
        "is_out_of_scope": True, 
        "justification": "Fallback due to invalid LLM output"
    }

# Rewrite generation
rephraser_chain = (
    repharser_prompt
    | repharser_llm
    | RunnableLambda(parse_rewrites)
)

generate_rewrites_chain = (
    RunnableAssign({
        "rewrites": lambda x: rephraser_chain.invoke({"query": x["query"]})
    })
    | RunnableAssign({
        "all_queries": lambda x: [x["query"]] + x["rewrites"]
    })
)

# Retrieval
retrieve_chain = RunnableLambda(hybrid_retrieve)
hybrid_chain = generate_rewrites_chain | retrieve_chain

# Validation
extract_validation_inputs = RunnableLambda(lambda x: {
    "query": x["query"],
    "contents": [c["content"] for c in x["chunks"]]
})

validation_chain = (
    extract_validation_inputs
    | relevance_prompt
    | relevance_llm
    | RunnableLambda(safe_json_parse)
)

# Answer Generation
def prepare_answer_inputs(x: Dict) -> Dict:
    context = KRISHNA_BIO if x["validation"]["is_out_of_scope"] else "\n\n".join(
        [x["chunks"][i-1]["content"] for i in x["validation"]["valid_chunks"]])
    
    return {
        "query": x["query"],
        "profile": KRISHNA_BIO,
        "context": context,
        "use_fallback": x["validation"]["is_out_of_scope"]
    }

select_and_prompt = RunnableLambda(lambda x: 
    answer_prompt_fallback.invoke(x) if x["use_fallback"]
    else answer_prompt_relevant.invoke(x))

answer_chain = (
    prepare_answer_inputs
    | select_and_prompt
    | relevance_llm
)

# Full Pipeline
full_pipeline = (
    hybrid_chain
    | RunnableAssign({"validation": validation_chain})
    | RunnableAssign({"answer": answer_chain})
)

import gradio as gr

def chat_interface(message, history):
    inputs = {
        "query": message,
        "all_queries": [message],
        "all_texts": all_chunks,
        "k_per_query": 3,
        "alpha": 0.7,
        "vectorstore": vectorstore,
        "full_document": "",
    }
    response = ""
    for chunk in full_pipeline.stream(inputs):
        if isinstance(chunk, str):
            response += chunk
            yield response
        elif isinstance(chunk, dict) and "answer" in chunk:
            response += chunk["answer"]
            yield response

with gr.Blocks(css="""
     html, body, .gradio-container {
        height: 100%;
        margin: 0;
        padding: 0;
    }
    .gradio-container {
        width: 90%;
        max-width: 1000px;
        margin: 0 auto;
        padding: 1rem;
    }

    .chatbox-container {
        display: flex;
        flex-direction: column;
        height: 95%;
    }

    .chatbot {
        flex: 1;
        overflow-y: auto;
        min-height: 500px;
    }

    .textbox {
        margin-top: 1rem;
    }
    #component-523 {
        height: 98%;
    }
""") as demo:
    with gr.Column(elem_classes="chatbox-container"):
        gr.Markdown("## 💬 Ask Krishna's AI Assistant")
        gr.Markdown("💡 Ask anything about Krishna Vamsi Dhulipalla")
        chatbot = gr.Chatbot(elem_classes="chatbot")
        textbox = gr.Textbox(placeholder="Ask a question about Krishna...", elem_classes="textbox")

        gr.ChatInterface(
            fn=chat_interface,
            chatbot=chatbot,
            textbox=textbox,
            examples=[
                "What are Krishna's research interests?",
                "Where did Krishna work?",
                "What did he study at Virginia Tech?"
            ],
        )

demo.launch()