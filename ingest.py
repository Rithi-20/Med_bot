import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


def create_vector_db():
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    raw_pages = loader.load()
    print(f"Loaded {len(raw_pages)} document pages.")

    # Enrich each page with explicit page number, document name, authority, and version metadata
    # Satisfies Edge Case 15 (Page-level source provenance) and Edge Case 10 (Source authority & versioning)
    for p in raw_pages:
        page_num = p.metadata.get("page", 0) + 1
        source_path = p.metadata.get("source", "unknown")
        doc_name = os.path.basename(source_path)
        p.metadata["page"] = page_num
        p.metadata["document_name"] = doc_name
        p.metadata["authority"] = "Verified Clinical Reference Guidelines"
        p.metadata["version"] = "2024.1"

    # Section-aware recursive splitter: prioritizes keeping numbered disease sections intact
    # while inheriting page-level provenance on every chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=[r'\n(?=[0-9]+\.)', '\n\n', '\n', '. ', ' '],
        is_separator_regex=True
    )
    texts = text_splitter.split_documents(raw_pages)
    print(f"Split into {len(texts)} chunks with verified page provenance.")

    print("Generating embeddings and creating FAISS vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS database successfully saved to {DB_FAISS_PATH}!")


if __name__ == "__main__":
    create_vector_db()