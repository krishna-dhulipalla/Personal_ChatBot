import os
import json
import re
import hashlib
import gradio as gr
from functools import partial
import concurrent.futures
from collections import defaultdict
from pathlib import Path
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
FAISS_PATH = "faiss_store/v30_600_150"
CHUNKS_PATH = "all_chunks.json"

if not Path(FAISS_PATH).exists():
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

if not Path(CHUNKS_PATH).exists():
    raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

KRISHNA_BIO = """Krishna Vamsi Dhulipalla is a 2024 graduate of the M.Eng program in Computer Science at Virginia Tech, with over 3 years of experience across data engineering, machine learning research, and real-time analytics. He specializes in building scalable data systems and intelligent LLM-powered applications, with strong expertise in Python, PyTorch, Hugging Face Transformers, and end-to-end ML pipelines.

He has led projects involving retrieval-augmented generation (RAG), feature selection for genomic classification, fine-tuning domain-specific LLMs (e.g., DNABERT, HyenaDNA), and real-time forecasting systems using Kafka, Spark, and Airflow. His cloud proficiency spans AWS (S3, SageMaker, ECS, CloudWatch), GCP (BigQuery, Cloud Composer), and DevOps tools like Docker, Kubernetes, and MLflow.

Krishna’s research has focused on genomic sequence modeling, transformer optimization, MLOps automation, and cross-domain generalization. He has published work in bioinformatics and machine learning applications for circadian transcription prediction and transcription factor binding.

He holds certifications in NVIDIA’s RAG Agents with LLMs, Google Cloud Data Engineering, and AWS ML Specialization. Krishna is passionate about scalable LLM infrastructure, data-centric AI, and domain-adaptive ML solutions — combining deep technical expertise with real-world engineering impact."""

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

# Initialize the knowledge base
knowledge_base = KnowledgeBase()


# LLMs
# repharser_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3") | StrOutputParser()
repharser_llm = ChatNVIDIA(model="microsoft/phi-3-mini-4k-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()
relevance_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
answer_llm = ChatOpenAI(
    model="gpt-4o",              
    temperature=0.3,             
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()] 
) | StrOutputParser()


# Prompts
repharser_prompt = ChatPromptTemplate.from_template(
    "Rewrite the question below in 3 different ways to help retrieve related information. Vary tone, style, and phrasing, but keep the meaning the same."
    "Question: {query}"
    "\n\nRewrites:"
    "1."
    "2."
    "3."
)

relevance_prompt = ChatPromptTemplate.from_template("""
You are Krishna's personal AI assistant classifier.

Your job is to decide whether a user's question can be meaningfully answered using the provided document chunks **or** relevant user memory.

Return a JSON object:
- "is_out_of_scope": true if the chunks and memory cannot help answer the question
- "justification": a short sentence explaining your decision

---

Special instructions:

✅ Treat short or vague queries like "yes", "tell me more", "go on", or "give me" as follow-up prompts. 
Assume the user is asking for **continuation** of the previous assistant response or follow-ups stored in memory. Consider that context as *in-scope*.

✅ Also consider if the user's question can be answered using stored memory (like their name, company, interests, or last follow-up topics).

Do NOT classify these types of queries as "out of scope".

Only mark as out-of-scope if the user asks something truly unrelated to both:
- Krishna's background
- Stored user memory

---

Examples:

Q: "Tell me more"
Chunks: previously retrieved info about Krishna's ML tools  
Memory: User previously asked about PyTorch and ML pipelines

Output:
{{
  "is_out_of_scope": false,
  "justification": "User is requesting a follow-up to a valid context, based on prior conversation"
}}

Q: "What is Krishna's Hogwarts house?"
Chunks: None about fiction  
Memory: User hasn't mentioned fiction/fantasy

Output:
{{
  "is_out_of_scope": true,
  "justification": "The question is unrelated to Krishna or user context"
}}

---

Now your turn.

User Question:
"{query}"

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
    "Example:\n"
    "**Q: What work experience does Krishna have?**\n"
    "**A:**\n"
    "**🔧 Work Experience Overview**\n"
    "**1. UJR Technologies** – Migrated batch ETL to real-time (Kafka/Spark), Dockerized services, and optimized Snowflake queries.\n"
    "**2. Virginia Tech** – Built real-time IoT forecasting pipeline (10K sensors, GPT-4), achieving 91% accuracy and 15% energy savings.\n\n"
    "_Would you like to dive into Krishna’s cloud deployment work using SageMaker and MLflow?_\n\n"
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
    return [line.strip("0123456789. ").strip() for line in lines if line.strip()][:3]

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
    "contents": [c["content"] for c in x["chunks"]],
    "memory": knowledge_base.json()
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
        "memory": knowledge_base.json()
    }

select_and_prompt = RunnableLambda(lambda x: 
    answer_prompt_fallback.invoke(x) if x["use_fallback"]
    else answer_prompt_relevant.invoke(x))

answer_chain = (
    prepare_answer_inputs
    | select_and_prompt
    | relevance_llm
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
    llm=relevance_llm,            
    prompt=parser_prompt        
)

def update_kb_after_answer(data: dict):
    try:
        kb_input = {
            "know_base": knowledge_base.json(),
            "input": data["query"],
            "output": data["answer"]
        }

        new_kb = knowledge_extractor.invoke(kb_input)
        knowledge_base.__dict__.update(new_kb.__dict__)  # update in place

        # Optional: print or log updated KB
        # print("✅ Knowledge base updated:", knowledge_base.dict())

    except Exception as e:
        print("❌ Failed to update knowledge base:", str(e))

    return data  # Return unchanged so answer can flow forward


update_kb_chain = RunnableLambda(update_kb_after_answer)

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
    full_response = ""
    collected = None

    for chunk in full_pipeline.stream(inputs):
        if isinstance(chunk, dict) and "answer" in chunk:
            full_response += chunk["answer"]
            collected = chunk  # store result for memory update
            yield full_response
        elif isinstance(chunk, str):
            full_response += chunk
            yield full_response

    # After yielding the full response, run knowledge update in background
    if collected:
        update_kb_after_answer({
            "query": message,
            "answer": full_response
        })
            
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