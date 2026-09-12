from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


def create_vector_db():
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    raw_pages = loader.load()
    print(f"Loaded {len(raw_pages)} document pages.")

    # Group pages by PDF document so headings across page breaks are not severed
    docs_by_source = {}
    for doc in raw_pages:
        src = doc.metadata.get("source", "unknown")
        if src not in docs_by_source:
            docs_by_source[src] = []
        docs_by_source[src].append(doc)

    merged_docs = []
    for src, pages in docs_by_source.items():
        full_text = ""
        for p in pages:
            page_num = p.metadata.get("page", 0) + 1
            full_text += f"\n\n[Page {page_num}]\n" + p.page_content
        merged_docs.append(Document(page_content=full_text, metadata={"source": src}))

    # Section-aware recursive splitter: prioritizes keeping numbered disease sections intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=[r'\n(?=[0-9]+\.)', '\n\n', '\n', '. ', ' '],
        is_separator_regex=True
    )
    texts = text_splitter.split_documents(merged_docs)
    print(f"Split into {len(texts)} intact disease chunks.")

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