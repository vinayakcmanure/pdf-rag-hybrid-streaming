import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.rag_engine import RAGEngine

app = FastAPI()
rag = RAGEngine()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # ✅ SAVE CORRECTLY AS BINARY
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()   # ✅ IMPORTANT
            tmp.write(contents)
            tmp_path = tmp.name

        print("✅ File saved at:", tmp_path)

        rag.process_pdf(tmp_path)

        return {"message": "✅ PDF processed successfully"}

    except Exception as e:
        print("❌ Upload error:", str(e))
        return {"message": f"Error: {str(e)}"}


@app.post("/chat-stream")
async def chat_stream(query: str):

    def generate():
        docs = rag.hybrid_search(query)
        docs = rag.rerank(query, docs)

        for token in rag.stream_answer(query, docs):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")