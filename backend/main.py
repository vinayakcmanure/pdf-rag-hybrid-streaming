import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.rag_engine import RAGEngine

app = FastAPI()

# ✅ Initialize RAG engine
rag = RAGEngine()


# ---------------------------
# ✅ Upload PDF
# ---------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # ✅ Validate file type
        if not file.filename.endswith(".pdf"):
            return {"message": "❌ Only PDF files are supported"}

        # ✅ Save file temporarily (binary safe)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        print("✅ File saved at:", tmp_path)

        # ✅ Process PDF
        rag.process_pdf(tmp_path)

        return {"message": "✅ PDF processed successfully"}

    except Exception as e:
        print("❌ Upload error:", str(e))
        return {"message": f"❌ Upload Error: {str(e)}"}


# ---------------------------
# ✅ Chat Streaming API
# ---------------------------
@app.post("/chat-stream")
async def chat_stream(query: str):
    try:
        if not query:
            return StreamingResponse(
                iter(["❌ Query cannot be empty"]),
                media_type="text/plain"
            )

        def generate():
            try:
                docs = rag.hybrid_search(query)
                docs = rag.rerank(query, docs)

                for token in rag.stream_answer(query, docs):
                    yield token

            except Exception as e:
                yield f"❌ Chat Error: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        print("❌ Chat endpoint error:", str(e))
        return StreamingResponse(
            iter([f"❌ Server Error: {str(e)}"]),
            media_type="text/plain"
        )