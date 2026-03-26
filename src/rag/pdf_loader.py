import re
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(path: str) -> str:
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
