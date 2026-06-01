import tempfile
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from rag_engine import RAGEngine

app = FastAPI()
rag = RAGEngine()

@app.post('/upload')
async def upload(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(await file.read())
        path = tmp.name
    rag.process_pdf(path)
    return {'message': '✅ PDF processed'}

@app.post('/chat-stream')
async def chat_stream(query: str):
    def generate():
        docs = rag.hybrid_search(query, k=5)
        docs = rag.rerank(query, docs)
        for token in rag.stream_answer(query, docs):
            yield token
    return StreamingResponse(generate(), media_type='text/plain')
