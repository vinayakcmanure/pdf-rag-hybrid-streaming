import requests
import gradio as gr

API_URL = 'http://localhost:8000'

def upload_pdf(file):
    files = {'file': file}
    res = requests.post(f"{API_URL}/upload", files=files)
    return res.json()['message']


def chat_stream(message, history):
    response = requests.post(
        f"{API_URL}/chat-stream",
        params={'query': message},
        stream=True
    )
    partial = ''
    for chunk in response.iter_content(chunk_size=20):
        if chunk:
            text = chunk.decode('utf-8')
            partial += text
            yield partial + '▌'

with gr.Blocks() as demo:
    gr.Markdown('# 🚀 RAG Hybrid Streaming App')
    file = gr.File(label='Upload PDF')
    btn = gr.Button('Process')
    status = gr.Textbox()
    chat = gr.ChatInterface(fn=chat_stream)
    btn.click(upload_pdf, inputs=file, outputs=status)

demo.launch()
