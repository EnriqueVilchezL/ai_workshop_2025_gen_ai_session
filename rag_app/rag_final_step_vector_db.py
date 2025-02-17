from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings

embedding_model = "llama3.1:8b"

def load_documents_from_my_sources(sources_path: Path) -> list:
    source_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    for source in source_documents:
        file_path = sources_path / source["file"]
        match source["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue
        documents = loader.load()
        for doc in documents:
            doc.metadata = {"source": source["file"].name}
        docs_parsed.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )

    return text_splitter.split_documents(docs_parsed)


def create_vector_db() -> Chroma:
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / "my_sources"
    persistent_directory = current_dir / "db"

    embeddings = OllamaEmbeddings(model=embedding_model)
    if persistent_directory.exists():
        print("Using existing db")
        return Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("Persistent directory does not exist. Initializing vector store...")
    
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )
    docs = load_documents_from_my_sources(sources_path)
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persistent_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )

