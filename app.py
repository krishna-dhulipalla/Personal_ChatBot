import os
import json
import re
import hashlib
import os
import json
import re
import hashlib
import gradio as gr
from functools import partial
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, List, Literal, Type
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.style import Style
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

#dotenv_path = os.path.join(os.getcwd(), ".env")
#load_dotenv(dotenv_path)
#api_key = os.getenv("NVIDIA_API_KEY")
#os.environ["NVIDIA_API_KEY"] = api_key
load_dotenv()
api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("🚨 NVIDIA_API_KEY not found in environment! Please add it in Hugging Face Secrets.")

# Constants
FAISS_PATH = "faiss_store/v64_600-150"
CHUNKS_PATH = "all_chunks.json"

if not Path(FAISS_PATH).exists():
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

if not Path(CHUNKS_PATH).exists():
    raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

KRISHNA_BIO = """Krishna Vamsi Dhulipalla completed masters in Computer Science at Virginia Tech, awarded degree in december 2024, with over 3 years of experience across data engineering, machine learning research, and real-time analytics. He specializes in building scalable data systems and intelligent LLM-powered applications, with strong expertise in Python, PyTorch, Hugging Face Transformers, and end-to-end ML pipelines.
He has led projects involving retrieval-augmented generation (RAG), feature selection for genomic classification, fine-tuning domain-specific LLMs (e.g., DNABERT, HyenaDNA), and real-time forecasting systems using Kafka, Spark, and Airflow. His cloud proficiency spans AWS (S3, SageMaker, ECS, CloudWatch), GCP (BigQuery, Cloud Composer), and DevOps tools like Docker, Kubernetes, and MLflow.
Krishna’s research has focused on genomic sequence modeling, transformer optimization, MLOps automation, and cross-domain generalization. He has published work in bioinformatics and machine learning applications for circadian transcription prediction and transcription factor binding.
He holds certifications in NVIDIA’s RAG Agents with LLMs, Google Cloud Data Engineering, and AWS ML Specialization. Krishna is passionate about scalable LLM infrastructure, data-centric AI, and domain-adaptive ML solutions — combining deep technical expertise with real-world engineering impact.
\n\n
Beside carrer, Krishna loves hiking, cricket, and exploring new technologies. He is big fan of Marvel Movies and Space exploration.
"""

def initialize_console():
    console = Console()
    base_style = Style(color="#76B900", bold=True)
    return partial(console.print, style=base_style)

pprint = initialize_console()

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

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

# Define the KnowledgeBase model
class KnowledgeBase(BaseModel):
    user_name: str = Field('unknown', description="The name of the user chatting with Krishna's assistant, or 'unknown' if not provided")
    company: Optional[str] = Field(None, description="The company or organization the user is associated with, if mentioned")
    last_input: str = Field("", description="The most recent user question or message")
    last_output: str = Field("", description="The most recent assistant response to the user")
    summary_history: List[str] = Field(default_factory=list, description="Summarized conversation history over turns")
    recent_interests: List[str] = Field(default_factory=list, description="User's recurring interests or topics they ask about, e.g., 'LLMs', 'Krishna's research', 'career advice'")
    last_followups: List[str] = Field(default_factory=list, description="List of follow-up suggestions from the last assistant response")
    tone: Optional[Literal['formal', 'casual', 'playful', 'direct', 'uncertain']] = Field(None, description="Inferred tone or attitude from the user based on recent input")

    def dump_truncated(self, max_len: int = 500):
        memory = self.dict()
        if len(memory["last_input"]) > max_len:
            memory["last_input"] = memory["last_input"][:max_len] + "..."
        if len(memory["last_output"]) > max_len:
            memory["last_output"] = memory["last_output"][:max_len] + "..."
        return memory

# Initialize the knowledge base
# knowledge_base = KnowledgeBase()
user_kbs = {}
kb_lock = Lock()

# LLMs
# repharser_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3") | StrOutputParser()
#repharser_llm = ChatNVIDIA(model="microsoft/phi-3-mini-4k-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()
rephraser_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3") | StrOutputParser()
relevance_llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct") | StrOutputParser()
answer_llm = ChatOpenAI(
    model="gpt-4o",              
    temperature=0.3,             
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True
) | StrOutputParser()


# Prompts
repharser_prompt = ChatPromptTemplate.from_template(
    "You are a smart retrieval assistant helping a search engine understand user intent more precisely.\n\n"
    "Your job is to rewrite the user's message into a clearer, more descriptive query for information retrieval.\n\n"
    "Context:\n"
    "- The user may sometimes respond with short or vague messages like 'B', 'yes', or 'tell me more'.\n"
    "- In such cases, refer to the user's previous assistant message or `last_followups` list to understand the actual intent.\n"
    "- Expand their reply based on that context to create a full meaningful query.\n\n"
    "User Query:\n{query}\n\n"
    "Last Follow-up Suggestions:\n{memory}\n\n"
    "Guidelines:\n"
    "- Expand abbreviations or implied selections\n"
    "- Reconstruct full intent if the query is a reply to an earlier assistant suggestion\n"
    "- Rephrase using domain-specific terms (e.g., ML, infrastructure, research, deployment)\n"
    "- Focus on maximizing retrievability via keyword-rich formulation\n\n"
    "- Prioritize clarity and retrievability over stylistic variation\n\n"
    "Expanded Rewrite:\n1."
)

relevance_prompt = ChatPromptTemplate.from_template("""
You are Krishna's personal AI assistant classifier and chunk reranker.

Your job has two goals:
1. Classify whether a user's question can be meaningfully answered using the retrieved document chunks or user memory.
2. If it can, rerank the chunks from most to least relevant to the question.

Return a JSON object:
- "is_out_of_scope": true if the **rewritten query**, original query, and memory offer no path to answer the user’s intent
- "justification": short explanation of your decision
- "reranked_chunks": a list of chunk indices ordered by decreasing relevance (only if in-scope)

---

Special Instructions:

✅ If the user input is vague, short, or a follow-up (e.g., "yes", "A", "B", "go on", "sure"), check:
• If the assistant previously showed suggestions or follow-up questions (in memory → `last_followups`)
• If the rewritten query adds meaningful context (e.g., "B" → "Tell me more about Data-Centric AI")

If **any of the above** resolve the intent, treat it as in-scope.

❌ Mark as out-of-scope only if:
- The query (even after rewriting) has no clear relevance to Krishna's profile or user memory
- There are no helpful document chunks or memory fields to answer it

🚫 Do not infer meaning through metaphor or vague similarity — only use concrete, literal context.

---

Examples:

Q: "B"
Rewritten Query: "Tell me more about Data-Centric AI for Real-Time Analytics"
last_followups: [ ... contains that option ... ]
Memory: user showed interest in analytics pipelines

Output:
{{
  "is_out_of_scope": false,
  "justification": "User is selecting a previous assistant suggestion",
  "reranked_chunks": [0, 2, 1]
}}

Q: "What is Krishna's Hogwarts house?"
Chunks: none on fiction
Memory: no fantasy topics

Output:
{{
  "is_out_of_scope": true,
  "justification": "Fictional topic unrelated to Krishna or conversation"
}}

---

Now your turn.

Original User Question:
"{query}"

Rewritten Query (if available):
"{rewritten_query}"

Chunks:
{contents}

User Memory (Knowledge Base):
{memory}

Return ONLY the JSON object.
""")

answer_prompt_relevant = ChatPromptTemplate.from_template(
    "You are Krishna's personal AI assistant. Your job is to answer the user’s question clearly, thoroughly, and professionally using the provided context.\n"
    "Rather than copying sentences, synthesize relevant insights and explain them like a knowledgeable peer.\n\n"
    "Use relevant memory about the user to personalize the answer where appropriate.\n\n"
    "Krishna's Background:\n{profile}\n\n"
    "User Memory (Knowledge Base):\n{memory}\n\n"
    "Context:\n{context}\n\n"
    "Instructions:\n"
    "- Format your response in **Markdown** for readability.\n"
    "- Use **section headings with emojis** to organize the answer when helpful (e.g., 🔍 Overview, 🛠️ Tools Used, 📈 Real-World Impact).\n"
    "- Use bullet points or bold text to highlight tools, skills, or project names.\n"
    "- Add paragraph breaks between major ideas.\n"
    "- Keep the tone conversational and helpful — like a smart peer explaining something.\n"
    "- If the user asks about Krishna’s work experience, provide a **chronological summary** of his roles and key contributions (e.g., UJR, Virginia Tech).\n"
    "- You may use general knowledge to briefly explain tools (like PyTorch or Kafka), but **do not invent any new facts** about Krishna.\n"
    "- Avoid filler phrases, repetition, or generic praise (e.g., strengths) unless directly asked.\n"
    "- End with a friendly follow-up question (no subheading needed here).\n\n"
    "Now generate the answer for the following:\n\n"
    "User Question:\n{query}\n\n"
    "Answer:"
)

answer_prompt_fallback = ChatPromptTemplate.from_template(
    "You are Krishna’s personal AI assistant. The user asked a question unrelated to Krishna’s background.\n"
    "Respond with a touch of humor, then guide the conversation back to Krishna’s actual skills, experiences, or projects.\n\n"
    "Make it clear that everything you mention afterward comes from Krishna's actual profile.\n\n"
    "Krishna's Background:\n{profile}\n\n"
    "User Memory (Knowledge Base):\n{memory}\n\n"
    "User Question:\n{query}\n\n"
    "Your Answer:"
)

parser_prompt = ChatPromptTemplate.from_template(
    "You are Krishna's personal AI assistant, and your task is to maintain a memory of the user you're chatting with.\n"
    "You just received a new user message and provided a response.\n"
    "Please update the knowledge base using the schema below.\n\n"
    "{format_instructions}\n\n"
    "Previous Knowledge Base:\n{know_base}\n\n"
    "Latest Assistant Response:\n{output}\n\n"
    "Latest User Message:\n{input}\n\n"
    "Return ONLY the updated knowledge base JSON:\n"
    "If the assistant’s response includes follow-up suggestions or continuation prompts (like 'Would you like to learn more about...'), store them in the `last_followups` field."
)

# Helper Functions
def parse_rewrites(raw_response: str) -> list[str]:
    lines = raw_response.strip().split("\n")
    return [line.strip("0123456789. ").strip() for line in lines if line.strip()][:1]

def hybrid_retrieve(inputs, exclude_terms=None):
    bm25_retriever = inputs["bm25_retriever"]
    all_queries = inputs["all_queries"]
    bm25_retriever.k = inputs["k_per_query"]
    vectorstore = inputs["vectorstore"]
    alpha = inputs["alpha"]
    top_k = inputs.get("top_k", 30)
    k_per_query = inputs["k_per_query"]

    scored_chunks = defaultdict(lambda: {
        "vector_scores": [],
        "bm25_score": 0.0,
        "content": None,
        "metadata": None,
    })

    def process_subquery(subquery, k=k_per_query):
        vec_hits = vectorstore.similarity_search_with_score(subquery, k=k)
        bm_hits = bm25_retriever.invoke(subquery)

        vec_results = [
            (hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(), doc, score)
            for doc, score in vec_hits
        ]

        bm_results = [
            (hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(), doc, 1.0 / (rank + 1))
            for rank, doc in enumerate(bm_hits)
        ]

        return vec_results, bm_results

    # Process each subquery serially
    for subquery in all_queries:
        vec_results, bm_results = process_subquery(subquery)

        for key, doc, vec_score in vec_results:
            scored_chunks[key]["vector_scores"].append(vec_score)
            scored_chunks[key]["content"] = doc.page_content
            scored_chunks[key]["metadata"] = doc.metadata

        for key, doc, bm_score in bm_results:
            scored_chunks[key]["bm25_score"] += bm_score
            scored_chunks[key]["content"] = doc.page_content
            scored_chunks[key]["metadata"] = doc.metadata

    all_vec_means = [np.mean(v["vector_scores"]) for v in scored_chunks.values() if v["vector_scores"]]
    max_vec = max(all_vec_means) if all_vec_means else 1
    min_vec = min(all_vec_means) if all_vec_means else 0

    final_results = []
    for chunk in scored_chunks.values():
        vec_score = np.mean(chunk["vector_scores"]) if chunk["vector_scores"] else 0.0
        norm_vec = 0.5 if max_vec == min_vec else (vec_score - min_vec) / (max_vec - min_vec)
        bm25_score = chunk["bm25_score"] / len(all_queries)
        final_score = alpha * norm_vec + (1 - alpha) * bm25_score

        content = chunk["content"].lower()
        if final_score < 0.01 or len(content.strip()) < 40:
            continue

        final_results.append({
            "content": chunk["content"],
            "source": chunk["metadata"].get("source", ""),
            "final_score": float(round(final_score, 4))
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
    | rephraser_llm
    | RunnableLambda(parse_rewrites)
)

generate_rewrites_chain = (
    RunnableAssign({
        "rewrites": lambda x: rephraser_chain.invoke({"query": x["query"],
                                                       "memory": x["memory"]})
    })
    | RunnableAssign({
        "all_queries": lambda x: [x["query"]] + x["rewrites"],
        "rewritten_query": lambda x: x["rewrites"][0] if x["rewrites"] else x["query"]
    })
)

# Retrieval
retrieve_chain = RunnableLambda(hybrid_retrieve)
hybrid_chain = generate_rewrites_chain | retrieve_chain

# Validation
extract_validation_inputs = RunnableLambda(lambda x: {
    "query": x["query"],
    "rewritten_query": x.get("rewritten_query", x["query"]),
    "contents": "\n".join(
        f"Chunk #{i}: {chunk['content']}" for i, chunk in enumerate(x["chunks"])
    ),
    "memory": json.dumps(x["memory"])
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
        "use_fallback": x["validation"]["is_out_of_scope"],
        "memory": x["memory"] 
    }

select_and_prompt = RunnableLambda(lambda x: 
    answer_prompt_fallback.invoke(x) if x["use_fallback"]
    else answer_prompt_relevant.invoke(x))

answer_chain = (
    prepare_answer_inputs
    | select_and_prompt
    #| relevance_llm
    | answer_llm
)

def RExtract(pydantic_class: Type[BaseModel], llm, prompt):
    """
    Runnable Extraction module for updating Krishna Assistant's KnowledgeBase.
    Fills in a structured schema using PydanticOutputParser.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({
        'format_instructions': lambda x: parser.get_format_instructions()
    })

    def preparse(raw: str):
        # Clean malformed LLM outputs
        if '{' not in raw: raw = '{' + raw
        if '}' not in raw: raw = raw + '}'
        return (raw
                .replace("\\_", "_")
                .replace("\n", " ")
                .replace("\]", "]")
                .replace("\[", "[")
        )

    return instruct_merge | prompt | llm | RunnableLambda(preparse) | parser

knowledge_extractor = RExtract(
    pydantic_class=KnowledgeBase,
    llm=instruct_llm,            
    prompt=parser_prompt        
)

def get_knowledge_base(session_id: str) -> KnowledgeBase:
    """Get or create a knowledge base for a session"""
    with kb_lock:
        if session_id not in user_kbs:
            user_kbs[session_id] = KnowledgeBase()
        return user_kbs[session_id]

def update_knowledge_base(session_id: str, user_input: str, assistant_response: str):
    """Update the knowledge base for a specific session"""
    try:
        kb = get_knowledge_base(session_id)
        kb_input = {
            "know_base": kb.model_dump_json(),
            "input": user_input,
            "output": assistant_response
        }
        new_kb = knowledge_extractor.invoke(kb_input)
        with kb_lock:
            user_kbs[session_id] = new_kb
        print(f"✅ KNOWLEDGE BASE UPDATED FOR SESSION {session_id}")
    except Exception as e:
        print(f"❌ KNOWLEDGE BASE UPDATE FAILED: {str(e)}")
        
def reorder_chunks_if_needed(inputs):
    validation = inputs.get("validation", {})
    chunks = inputs.get("chunks", [])

    if not validation.get("is_out_of_scope", True) and "reranked_chunks" in validation:
        try:
            ranked_indices = validation["reranked_chunks"]
            inputs["chunks"] = [chunks[i] for i in ranked_indices if i < len(chunks)]
        except Exception as e:
            print("⚠️ Failed to reorder chunks:", e)

    return inputs

# New chain to preserve memory through the pipeline
preserve_memory_chain = RunnableLambda(lambda x: {
    **hybrid_chain.invoke(x),  
    "memory": x["memory"]     
})

# Full pipeline
full_pipeline = (
    preserve_memory_chain 
    | RunnableAssign({"validation": validation_chain}) 
    | RunnableLambda(reorder_chunks_if_needed)
    #| PPrint()
    | answer_chain
)

def chat_interface(message, history, request: gr.Request):
    """Modified chat interface with session support"""
    session_id = request.session_hash
    kb = get_knowledge_base(session_id)
    
    # Initialize inputs with session-specific KB
    inputs = {
        "query": message,
        "all_queries": [message],
        "all_texts": all_chunks,
        "k_per_query": 7,
        "alpha": 0.5,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "memory": json.dumps(kb.dump_truncated())
    }
    
    full_response = ""
    for chunk in full_pipeline.stream(inputs):
        if isinstance(chunk, str):
            full_response += chunk
            yield full_response
    
    # Update KB after response
    if full_response:
        update_knowledge_base(session_id, message, full_response) 

demo = gr.ChatInterface(
    fn=chat_interface,
    title="💬 Ask Krishna's AI Assistant",
    css= """
    html, body {
        margin: 0;
        padding: 0;
        overflow-x: hidden; /* prevent horizontal scrollbars on body */
    }
    .column:nth-child(2) {
        max-width: 800px;
        margin: 0 auto;
        width: 100%;
    }
    .float {
        display: none;
    }
    .bubble-wrap {
        background: #0f0f11 !important;
    }
    .gr-group {
        border-radius: 2rem !important;
    }
    .flex {
        border: none !important;
    }
    footer {
        display: none !important;
    }
    ::-webkit-scrollbar {
    width: 1px;
    height: 1px;
}
 .gradio-container{
        max-width: 1000px !important;
        margin: 0 auto;
        width:100%;
    }

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background-color: rgba(255, 255, 255, 0.05); /* very light thumb */
    border-radius: 10px;
}

/* Scrollbar - Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.05) transparent;
}
    """,
    examples=[
        "Give me an overview of Krishna Vamsi Dhulipalla's work experience across different roles?",
        "What programming languages and tools does Krishna use for data science?",
        "Can this chatbot tell me what Krishna's chatbot architecture looks like and how it works?"
    ]
)
# Launch with request support
demo.queue()
demo.launch(
    max_threads=4,
    prevent_thread_lock=True,
    debug=True
)