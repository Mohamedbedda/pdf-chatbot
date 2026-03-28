import re
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


def extract_text_from_pdf(path: str) -> str:
    try:
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file, it may be corrupted : {str(e)} !")

    if not text or not text.strip():
        raise ValueError("PDF appears to be empty or unreadable !")

    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    # Filter out empty or whitespace-only chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        raise ValueError("No valid text chunks could be created from PDF")
    return chunks
