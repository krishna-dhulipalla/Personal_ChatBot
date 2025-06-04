import os
import re
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# === UTILS ===
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

def fix_json_text(text):
    # Normalize quotes and extract clean JSON
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text

def enrich_chunk_with_llm(text, llm):
    prompt = f"""You're a helpful assistant optimizing document retrieval.

            Every document you see is about Krishna Vamsi Dhulipalla.

            Here’s a document chunk:
            {text}

            1. Summarize the key content of this chunk in 1–2 sentences, assuming the overall context is about Krishna.
            2. Generate 3 natural-language questions that a user might ask to which this chunk would be a relevant answer, focusing on Krishna-related topics.

            Respond in JSON:
            {{
            "summary": "...",
            "synthetic_queries": ["...", "...", "..."]
            }}"""

    response = llm.invoke(prompt)
    content = getattr(response, "content", "").strip()

    if not content:
        raise ValueError("⚠️ LLM returned empty response")

    fixed = fix_json_text(content)
    try:
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\n--- Raw Output ---\n{content}")

# === MAIN FUNCTION ===
def create_faiss_store(
    md_dir="./personal_data",
    chunk_size=600,
    chunk_overlap=150,
    persist_dir="./faiss_store",
    chunk_save_path="all_chunks.json",
    llm=None
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n- ", "\n", ". ", " "],
        keep_separator=True,
        length_function=len,  # Consider switching to tokenizer-based later
        is_separator_regex=False
    )

    docs, all_chunks, failed_chunks = [], [], []

    for md_file in Path(md_dir).glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                continue
            content = re.sub(r'\n#+(\w)', r'\n# \1', content)
            docs.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "header": content.split('\n')[0]
                }
            })

    for doc in docs:
        try:
            chunks = splitter.split_text(doc["content"])
        except Exception as e:
            print(f"❌ Error splitting {doc['metadata']['source']}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 50:
                continue

            chunk_id = f"{doc['metadata']['source']}_#{i}_{hash_text(chunk)}"
            metadata = {
                **doc["metadata"],
                "chunk_id": chunk_id,
                "has_header": chunk.startswith("#"),
                "word_count": len(chunk.split())
            }

            try:
                print("🔍 Processing chunk:", chunk_id)
                enriched = enrich_chunk_with_llm(chunk, llm)
                summary = enriched.get("summary", "")
                questions = enriched.get("synthetic_queries", [])

                metadata.update({
                    "summary": summary,
                    "synthetic_queries": questions
                })

                enriched_text = (
                    f"{chunk}\n\n"
                    f"---\n"
                    f"🔹 Summary:\n{summary}\n\n"
                    f"🔸 Related Questions:\n" + "\n".join(f"- {q}" for q in questions)
                )

                all_chunks.append({
                    "text": enriched_text,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"⚠️ LLM failed for {chunk_id}: {e}")
                failed_chunks.append(f"{chunk_id} → {str(e)}")

    print(f"✅ Markdown files processed: {len(docs)}")
    print(f"✅ Chunks created: {len(all_chunks)} | ⚠️ Failed: {len(failed_chunks)}")

    # Save enriched chunks
    with open(chunk_save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved enriched chunks → {chunk_save_path}")

    os.makedirs(persist_dir, exist_ok=True)
    version_tag = f"v{len(all_chunks)}_{chunk_size}_{chunk_overlap}"
    save_path = os.path.join(persist_dir, version_tag)
    os.makedirs(save_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_texts(
        texts=[chunk["text"] for chunk in all_chunks],
        embedding=embeddings,
        metadatas=[chunk["metadata"] for chunk in all_chunks]
    )
    vector_store.save_local(save_path)

    print(f"✅ FAISS index saved at: {save_path}")
    avg_len = sum(len(c['text']) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    print(f"📊 Stats → Chunks: {len(all_chunks)} | Avg length: {avg_len:.1f} characters")

    if failed_chunks:
        with open("failed_chunks.txt", "w") as f:
            for line in failed_chunks:
                f.write(line + "\n")
        print("📝 Failed chunk IDs saved to failed_chunks.txt")
        
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)
api_key = os.getenv("NVIDIA_API_KEY")
os.environ["NVIDIA_API_KEY"] = api_key
# Initialize the model
llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")

create_faiss_store(
    md_dir="./personal_data",
    chunk_size=600,
    chunk_overlap=150,
    persist_dir="./faiss_store",
    llm=llm
)




# 
# from langchain.text_splitter import (
#     RecursiveCharacterTextSplitter,
#     MarkdownHeaderTextSplitter
# )
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from transformers import AutoTokenizer
# from pathlib import Path
# import os
# from typing import List

# def prepare_vectorstore(
#     base_path: str,
#     faiss_path: str,
#     use_markdown_headers: bool = True,
#     chunk_size: int = 600,
#     chunk_overlap: int = 150,
#     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
#     verbose: bool = True
# ) -> FAISS:
#     docs = []
#     for md_file in Path(base_path).glob("*.md"):
#         with open(md_file, "r", encoding="utf-8") as f:
#             content = f.read()
#         metadata = {
#             "source": md_file.name,
#             "file_type": "markdown",
#             "created_at": md_file.stat().st_ctime
#         }
#         docs.append(Document(page_content=content, metadata=metadata))

#     # Optional Markdown-aware splitting
#     if use_markdown_headers:
#         header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
#             ("#", "h1"), ("##", "h2"), ("###", "h3")
#         ])
#         structured_chunks = []
#         for doc in docs:
#             splits = header_splitter.split_text(doc.page_content)
#             for chunk in splits:
#                 chunk.metadata.update(doc.metadata)
#                 structured_chunks.append(chunk)
#     else:
#         structured_chunks = docs

#     # Tokenizer-based recursive splitting
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     recursive_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=lambda text: len(tokenizer.encode(text)),
#         separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
#     )

#     final_chunks: List[Document] = []
#     for chunk in structured_chunks:
#         sub_chunks = recursive_splitter.split_text(chunk.page_content)
#         for i, sub in enumerate(sub_chunks):
#             final_chunks.append(Document(
#                 page_content=sub,
#                 metadata={**chunk.metadata, "sub_chunk": i}
#             ))

#     if verbose:
#         print(f"✅ Total chunks after splitting: {len(final_chunks)}")
#         print(f"📁 Storing to: {faiss_path}")

#     embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#     vectorstore = FAISS.from_documents(final_chunks, embedding_model)
#     vectorstore.save_local(faiss_path)

#     if verbose:
#         print(f"✅ FAISS vectorstore saved at: {os.path.abspath(faiss_path)}")

#     return vectorstore

# vectorstore = prepare_vectorstore(
#     base_path="./personal_data",
#     faiss_path="krishna_vectorstore_hybrid",
#     use_markdown_headers=True,
#     chunk_size=600,
#     chunk_overlap=150,
#     verbose=True
# )
