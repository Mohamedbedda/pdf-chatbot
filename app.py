import gradio as gr

from src.config import settings
from src.state import AppState
from src.rag.pdf_loader import extract_text_from_pdf, chunk_text
from src.rag.retriever import build_indexes, retrieve
from src.rag.generator import generate_answer

state = AppState()

def upload_pdf(pdf_file):
    """
    Triggered when user uploads a PDF.
    Extracts text, builds indexes.
    """
    if pdf_file is None:
        return "Please upload a valid PDF."

    text = extract_text_from_pdf(pdf_file.name)
    chunks = chunk_text(text)

    index, bm25 = build_indexes(chunks)

    state.chunks = chunks
    state.index = index
    state.bm25 = bm25
    state.pdf_loaded = True

    return "PDF loaded successfully !"


def chat_with_pdf(message, model_key):
    """
    Triggered when user submits a question.
    Runs full RAG pipeline and returns the answer.
    """
    if not state.pdf_loaded:
        return "Please upload a PDF first."

    contexts = retrieve(message, state.chunks, state.index, state.bm25)

    if contexts is None:
        answer = "Not found in the document!"
    else:
        answer = generate_answer(message, contexts, model_key)

    return answer

# UI
with gr.Blocks(title="PDF RAG Chatbot") as demo:
    gr.Markdown("# PDF RAG Chatbot")
    gr.Markdown("Upload a PDF, choose a model, and start asking questions.")

    with gr.Row():
        pdf_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            file_count="single",
        )
        model_dropdown = gr.Dropdown(
            choices=list(settings.GROQ_MODELS.keys()),
            value=list(settings.GROQ_MODELS.keys())[0],
            label="Model",
            interactive=True,
        )

    status = gr.Textbox(
        label="Status",
        interactive=False,
        placeholder="Upload a PDF to get started...",
    )

    msg = gr.Textbox(
        label="Your question",
        placeholder="Ask something about the PDF...",
    )

    answer_output = gr.Textbox(
        label="Answer",
        interactive=False,
        placeholder="Answer...",
    )


    # Events
    pdf_input.upload(
        fn=upload_pdf,
        inputs=pdf_input,
        outputs=status,
    )

    msg.submit(
        fn=chat_with_pdf,
        inputs=[msg, model_dropdown],
        outputs=answer_output,
    )


if __name__ == "__main__":
    demo.queue().launch(share=True)
