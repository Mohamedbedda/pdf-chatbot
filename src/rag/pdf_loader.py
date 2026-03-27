import re
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


def extract_text_from_pdf(path: str) -> str:
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)
