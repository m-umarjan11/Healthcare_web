from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from medical_rag import DB_FAISS_PATH, get_embedding_model


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"


def load_pdf_files(data_path: Path):
    pdf_files = sorted(data_path.rglob("*.pdf")) if data_path.exists() else []
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_path}. Add medical PDFs before building the vector DB."
        )

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    for document in documents:
        source = document.metadata.get("source")
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = (BASE_DIR / source_path).resolve()
            try:
                document.metadata["source"] = str(source_path.relative_to(BASE_DIR))
            except ValueError:
                document.metadata["source"] = source_path.name
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


def main():
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    if not text_chunks:
        raise ValueError("PDF files were loaded, but no text chunks were created.")

    DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(text_chunks, get_embedding_model())
    db.save_local(str(DB_FAISS_PATH))
    print(f"Loaded {len(documents)} PDF page(s).")
    print(f"Created {len(text_chunks)} searchable chunks.")
    print(f"Saved FAISS database to {DB_FAISS_PATH}.")


if __name__ == "__main__":
    main()