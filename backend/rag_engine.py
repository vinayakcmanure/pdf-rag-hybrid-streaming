import os

# ---- FIX SSL ISSUE ON WINDOWS ----
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ✅ LOCAL embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ---------------------------
# ✅ CONFIG
# ---------------------------
CHAT_MODEL = "openai.gpt-5.2"

print("✅ MODEL:", CHAT_MODEL)
print("✅ BASE URL:", os.getenv("CG_GENERATIVE_ENGINE_URL"))

client = OpenAI(
    base_url=os.getenv("CG_GENERATIVE_ENGINE_URL"),
    api_key=os.getenv("CG_GENERATIVE_ENGINE_API_KEY"),
)


# ---------------------------
# ✅ LOCAL EMBEDDINGS
# ---------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------
# ✅ RAG ENGINE
# ---------------------------
class RAGEngine:
    def __init__(self):
        self.vector_db = None
        self.bm25 = None
        self.documents = None

    # ---------------------------
    # ✅ PROCESS PDF
    # ---------------------------
    def process_pdf(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )

            self.documents = splitter.split_documents(docs)
            print(f"✅ Total chunks: {len(self.documents)}")

            # ✅ Convert to LangChain docs
            lc_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in self.documents
            ]

            # ✅ FILTER OUT LOW QUALITY CHUNKS
            filtered_docs = [
                doc for doc in lc_docs
                if len(doc.page_content.strip()) > 200
            ]

            print(f"✅ Filtered chunks: {len(filtered_docs)}")

            # ✅ FULL FAISS (NO LIMIT)
            self.vector_db = FAISS.from_documents(
                filtered_docs,
                embedding_model
            )

            # ✅ BM25 uses ALL docs
            tokenized = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)

        except Exception as e:
            print("❌ ERROR in process_pdf:", str(e))
            raise

    # ---------------------------
    # ✅ HYBRID SEARCH
    # ---------------------------
    def hybrid_search(self, query, k=12):
        if self.vector_db is None:
            return []

        vector_docs = self.vector_db.similarity_search(query, k=k)

        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        bm25_docs = [self.documents[i] for i in top_idx]

        # ✅ Combine + remove duplicates
        combined = vector_docs + bm25_docs
        unique_docs = {doc.page_content: doc for doc in combined}.values()

        return list(unique_docs)

    # ---------------------------
    # ✅ RERANK
    # ---------------------------
    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        ranked = list(zip(scores, docs))
        ranked.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in ranked[:top_k]]

    # ---------------------------
    # ✅ STREAM ANSWER
    # ---------------------------
    def stream_answer(self, query, docs):
        if not docs:
            yield "⚠️ No relevant information found."
            return

        # ✅ DEBUG → see what RAG retrieved
        print("\n🔍 RETRIEVED DOCUMENTS:\n")
        for d in docs:
            print("----")
            print(d.page_content[:300])

        context = "\n\n".join([d.page_content for d in docs])

        messages = [
            {
                "role": "system",
                "content": """Use the provided context to answer the question.

If the context is incomplete, still provide the best possible structured answer 
based on available information.

Do not invent facts."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]

        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True
            )
        except Exception as e:
            yield f"❌ LLM ERROR: {str(e)}"
            return

        # ✅ STREAM RESPONSE
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        # ✅ SHOW SOURCES
        pages = set(
            d.metadata.get("page")
            for d in docs if "page" in d.metadata
        )
        if pages:
            yield f"\n\n📄 Sources: Pages {sorted(pages)}"
