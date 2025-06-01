import os
import json
import re
import hashlib
import gradio as gr
import time
from functools import partial
import concurrent.futures
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
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#dotenv_path = os.path.join(os.getcwd(), ".env")
#load_dotenv(dotenv_path)
#api_key = os.getenv("NVIDIA_API_KEY")
#os.environ["NVIDIA_API_KEY"] = api_key
load_dotenv()
api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("🚨 NVIDIA_API_KEY not found in environment! Please add it in Hugging Face Secrets.")

# Constants
FAISS_PATH = "faiss_store/v30_600_150"
CHUNKS_PATH = "all_chunks.json"

if not Path(FAISS_PATH).exists():
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

if not Path(CHUNKS_PATH).exists():
    raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

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

bm25_retriever = BM25Retriever.from_texts(texts=all_texts, metadatas=metadatas)

# LLMs
repharser_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()
relevance_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment!")
answer_llm = ChatOpenAI(
    model="gpt-4-1106-preview",              
    temperature=0.3,             
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()] 
) | StrOutputParser()


# Prompts
repharser_prompt = ChatPromptTemplate.from_template(
    "Rewrite the question below in 4 diverse ways to retrieve semantically similar information.Ensure diversity in phrasings across style, voice, and abstraction:\n\nQuestion: {query}\n\nRewrites:"
)

relevance_prompt = ChatPromptTemplate.from_template("""
You are Krishna's personal AI assistant classifier.

Your job is to decide whether a user's question can be meaningfully answered using the provided document chunks.

Think carefully and return a JSON object with:
- "is_out_of_scope": true if none of the chunks contain information relevant to the question.
- "justification": a short sentence explaining your decision.

---

Rules:
- Chunks are snippets from Krishna’s resume, project history, and personal background.
- If none of the chunks contain evidence, examples, or details that directly help answer the question, mark it as out of scope.
- Do NOT rely on keyword matches. Use reasoning to decide whether the content actually addresses the question.

Examples:

Q: "What are Krishna's favorite movies?"
Chunks: Mostly about research, skills, and work experience.

Output:
{{
  "is_out_of_scope": true,
  "justification": "No chunk discusses Krishna's personal preferences like movies."
}}

Q: "What ML tools has Krishna used in projects?"
Chunks: Mentions PyTorch, Kafka, Hugging Face, Spark.

Output:
{{
  "is_out_of_scope": false,
  "justification": "Chunks mention tools Krishna used directly in his work."
}}

---

Now your turn.

User Question:
"{query}"

Chunks:
{contents}

Return only the JSON object.
""")


answer_prompt_relevant = ChatPromptTemplate.from_template(
    "You are Krishna's personal AI assistant. Your job is to answer the user’s question clearly and professionally using the provided context.\n"
    "Rather than copying sentences, synthesize relevant insights and explain them like a knowledgeable peer.\n\n"
    "Krishna's Background:\n{profile}\n\n"
    "Note: The context might include some unrelated or noisy information. Focus only on content that directly supports your answer.\n\n"
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
    "Respond with a touch of humor, then guide the conversation back to Krishna’s actual skills, experiences, or projects.\n\n"
    "Make it clear that everything you mention afterward comes from Krishna's actual profile.\n\n"
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
    bm25_retriever = inputs["bm25_retriever"]
    all_queries = inputs["all_queries"]
    bm25_retriever.k = inputs["k_per_query"]
    vectorstore = inputs["vectorstore"]
    alpha = inputs["alpha"]
    top_k = inputs.get("top_k", 15)
    k_per_query = inputs["k_per_query"]

    scored_chunks = defaultdict(lambda: {
        "vector_scores": [],
        "bm25_score": 0.0,
        "content": None,
        "metadata": None,
    })
    
    # Function to process each subquery
    def process_subquery(subquery, k_per_query=3):
        # Vector retrieval
        vec_hits = vectorstore.similarity_search_with_score(subquery, k=k_per_query)
        vec_results = []
        for doc, score in vec_hits:
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            vec_results.append((key, doc, score))
        
        # BM25 retrieval
        bm_hits = bm25_retriever.invoke(subquery)
        bm_results = []
        for rank, doc in enumerate(bm_hits):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            bm_score = 1.0 - (rank / k_per_query)
            bm_results.append((key, doc, bm_score))
            
        return vec_results, bm_results

     # Process subqueries in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subquery, q) for q in all_queries]
        for future in concurrent.futures.as_completed(futures):
            vec_results, bm_results = future.result()
            
            # Process vector results
            for key, doc, score in vec_results:
                scored_chunks[key]["vector_scores"].append(score)
                scored_chunks[key]["content"] = doc.page_content
                scored_chunks[key]["metadata"] = doc.metadata
                
            # Process BM25 results
            for key, doc, bm_score in bm_results:
                scored_chunks[key]["bm25_score"] += bm_score
                scored_chunks[key]["content"] = doc.page_content
                scored_chunks[key]["metadata"] = doc.metadata

    # Rest of the scoring and filtering logic remains the same
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
    try:
        if isinstance(s, str) and "is_out_of_scope" in s:
            return json.loads(s)
    except json.JSONDecodeError:
        pass
    return {
        "is_out_of_scope": True,
        "justification": "Fallback due to invalid or missing LLM output"
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
    | instruct_llm
    | RunnableLambda(safe_json_parse)
)

# Answer Generation
def prepare_answer_inputs(x: Dict) -> Dict:
    context = KRISHNA_BIO if x["validation"]["is_out_of_scope"] else "\n\n".join(
        [chunk["content"] for chunk in x["chunks"]]
    )

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
full_pipeline = hybrid_chain | RunnableAssign({"validation": validation_chain}) | answer_chain


def chat_interface(message, history):
    inputs = {
        "query": message,
        "all_queries": [message],
        "all_texts": all_chunks,
        "k_per_query": 3,
        "alpha": 0.7,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
    }
    response = ""
    for chunk in full_pipeline.stream(inputs):
        if isinstance(chunk, str):
            response += chunk
            yield response
        elif isinstance(chunk, dict) and "answer" in chunk:
            response += chunk["answer"]
            yield response
            
demo = gr.ChatInterface(
    fn=chat_interface,
    title="💬 Ask Krishna's AI Assistant",
    description="💡 Ask anything about Krishna Vamsi Dhulipalla",
    # examples=[
    #     "What are Krishna's research interests?",
    #     "Where did Krishna work?",
    #     "What did he study at Virginia Tech?"
    # ],
    theme="default"
)

if __name__ == "__main__":
    demo.launch(max_threads=4, prevent_thread_lock=True, debug=True)

# with gr.Blocks(css="""
#      html, body, .gradio-container {
#         height: 100%;
#         margin: 0;
#         padding: 0;
#     }
#     .gradio-container {
#         width: 90%;
#         max-width: 1000px;
#         margin: 0 auto;
#         padding: 1rem;
#     }

#     .chatbox-container {
#         display: flex;
#         flex-direction: column;
#         height: 95%;
#     }

#     .chatbot {
#         flex: 1;
#         overflow-y: auto;
#         min-height: 500px;
#     }

#     .textbox {
#         margin-top: 1rem;
#     }
#     #component-523 {
#         height: 98%;
#     }
# """) as demo:
#     with gr.Column(elem_classes="chatbox-container"):
#         gr.Markdown("## 💬 Ask Krishna's AI Assistant")
#         gr.Markdown("💡 Ask anything about Krishna Vamsi Dhulipalla")
#         chatbot = gr.Chatbot(elem_classes="chatbot", type="messages")
#         textbox = gr.Textbox(placeholder="Ask a question about Krishna...", elem_classes="textbox")

#         gr.ChatInterface(
#             fn=chat_interface,
#             chatbot=chatbot,
#             textbox=textbox,
#             # examples=[
#             #     "What are Krishna's research interests?",
#             #     "Where did Krishna work?",
#             #     "What did he study at Virginia Tech?"
#             # ],
#             type= "messages",
#         )

# if __name__ == "__main__":
#     # Add resource verification
#     print(f"FAISS path exists: {Path(FAISS_PATH).exists()}")
#     print(f"Chunks path exists: {Path(CHUNKS_PATH).exists()}")
#     print(f"Vectorstore type: {type(vectorstore)}")
#     print(f"All chunks count: {len(all_chunks)}")
    
#     # Launch the application
#     demo.launch(debug=True)