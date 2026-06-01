
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class RAGEngine:
    def __init__(self):
        self.vector_db = None
        self.bm25 = None
        self.documents = None

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        self.documents = splitter.split_documents(docs)
        self.vector_db = FAISS.from_documents(self.documents, embedding_model)
        tokenized = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query, k=5):
        vector_docs = self.vector_db.similarity_search(query, k=k)
        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [self.documents[i] for i in top_idx]
        combined = vector_docs + bm25_docs
        unique = {doc.page_content: doc for doc in combined}.values()
        return list(unique)

    def rerank(self, query, docs, top_k=5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = list(zip(scores, docs))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def stream_answer(self, query, docs):
        context = "

".join([d.page_content for d in docs])
        prompt = f"Answer using context:
{context}
Question:{query}"
        stream = llm.stream(prompt)
        for chunk in stream:
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
