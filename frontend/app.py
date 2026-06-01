import requests
import gradio as gr

# Backend URL
API_URL = "http://localhost:8000"


# ---------------------------
# Upload PDF (FIXED)
# ---------------------------
def upload_pdf(file):
    try:
        # ✅ Gradio returns a file object → use file.name
        with open(file.name, "rb") as f:
            files = {
                "file": (file.name, f, "application/pdf")
            }

            res = requests.post(
                f"{API_URL}/upload",
                files=files
            )

        print("STATUS:", res.status_code)
        print("RESPONSE:", res.text)

        # ✅ Safe JSON handling
        try:
            data = res.json()
            return data.get("message", "✅ PDF processed")
        except:
            return f"❌ Backend Error:\n{res.text}"

    except Exception as e:
        return f"❌ Upload Error: {str(e)}"


# ---------------------------
# Chat (Streaming)
# ---------------------------
def chat_stream(message, history):
    try:
        response = requests.post(
            f"{API_URL}/chat-stream",
            params={"query": message},
            stream=True
        )

        partial = ""

        for chunk in response.iter_content(chunk_size=20):
            if chunk:
                text = chunk.decode("utf-8")
                partial += text
                yield partial + "▌"

        # final output
        yield partial

    except Exception as e:
        yield f"❌ Chat Error: {str(e)}"


# ---------------------------
# UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 RAG Hybrid Streaming App")

    file = gr.File(label="Upload PDF")
    btn = gr.Button("Process")

    status = gr.Textbox(label="Status")

    chat = gr.ChatInterface(
        fn=chat_stream,
        title="💬 Chat with your PDF"
    )

    btn.click(upload_pdf, inputs=file, outputs=status)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    demo.launch()