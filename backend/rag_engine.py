import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ---------------------------
# ENV
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------
# MODELS
# ---------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Fast reranker (10x faster than LLM rerank)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------
# RAG ENGINE
# ---------------------------
class RAGEngine:
    def __init__(self):
        self.vector_db = None
        self.bm25 = None
        self.documents = None

    # ---------------------------
    # PROCESS PDF
    # ---------------------------
    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # ✅ Improved chunking for large documents (Trafikverket)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,     # larger chunks = better context
            chunk_overlap=200
        )

        self.documents = splitter.split_documents(docs)

        print(f"✅ Total chunks created: {len(self.documents)}")

        # ✅ Vector (semantic)
        self.vector_db = FAISS.from_documents(self.documents, embedding_model)

        # ✅ BM25 (keyword)
        tokenized = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    # ---------------------------
    # HYBRID SEARCH
    # ---------------------------
    def hybrid_search(self, query, k=8):
        # ✅ Semantic search
        vector_docs = self.vector_db.similarity_search(query, k=k)

        # ✅ Keyword search
        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        bm25_docs = [self.documents[i] for i in top_idx]

        # ✅ Combine + deduplicate
        combined = vector_docs + bm25_docs
        unique_docs = {doc.page_content: doc for doc in combined}.values()

        return list(unique_docs)

    # ---------------------------
    # RERANK (FAST, LOCAL)
    # ---------------------------
    def rerank(self, query, docs, top_k=5):
        pairs = [(query, doc.page_content) for doc in docs]

        scores = reranker.predict(pairs)

        ranked = list(zip(scores, docs))
        ranked.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in ranked[:top_k]]

    # ---------------------------
    # ANSWER GENERATION (NON-STREAM)
    # ---------------------------
    def generate_answer(self, query, docs):
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an expert assistant answering questions about a document.

STRICT RULES:
- Answer ONLY from the provided context
- Do NOT guess or hallucinate
- If answer is not present, say: "Not found in document"
- Be precise and concise

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt).content

        # ✅ Page references
        pages = set()
        for d in docs:
            if "page" in d.metadata:
                pages.add(d.metadata["page"])

        if pages:
            response += f"\n\n📄 Sources: Pages {sorted(pages)}"

        return response

    # ---------------------------
    # STREAMING ANSWER (ChatGPT style)
    # ---------------------------
    def stream_answer(self, query, docs):
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an expert assistant answering questions about a document.

STRICT RULES:
- Answer ONLY from the provided context
- Do NOT guess or hallucinate
- If answer is not present, say: "Not found in document"
- Be precise and concise

Context:
{context}

Question:
{query}
"""

        stream = llm.stream(prompt)

        partial = ""

        for chunk in stream:
            if hasattr(chunk, "content") and chunk.content:
                partial += chunk.content
                yield chunk.content

        # ✅ Add page references at the end
        pages = set()
        for d in docs:
            if "page" in d.metadata:
                pages.add(d.metadata["page"])

        if pages:
            yield f"\n\n📄 Sources: Pages {sorted(pages)}"