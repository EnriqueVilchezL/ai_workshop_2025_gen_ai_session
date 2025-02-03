from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

model = "llama3.1:8b" 
embedding_model = "llama3.1:8b"


def testing_llama() -> None:
    chat_model = ChatOllama(model=model)
    result = chat_model.invoke("Who was the first man on the moon?")
    print(result.content)


def create_vector_db() -> Chroma:
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / "my_sources"
    persistent_directory = current_dir / "db" / "antonio_db"

    # Ensure the sources directory exists
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )

    # Load documents
    pdf_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    for pdf in pdf_documents:
        file_path = sources_path / pdf["file"]
        match pdf["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue
        documents = loader.load()
        for doc in documents:
            doc.metadata = {"source": pdf["file"].name}
        docs_parsed.extend(documents)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs_parsed)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(model=embedding_model)
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating vector store ---")
    if persistent_directory.exists():
        print("Using existing db")
        db = Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        print("Persistent directory does not exist. Initializing vector store...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=str(persistent_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    return db


def testing_db() -> None:
    db = create_vector_db()
    print("\n--- Finished creating vector store ---")

    query = "what are Antonio's skills?"

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.01},
    )
    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print(f"Source: {doc.metadata['source']}\n")


testing_db()
