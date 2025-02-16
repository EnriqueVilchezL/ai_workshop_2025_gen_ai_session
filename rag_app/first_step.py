from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph


model = "llama3.1:8b"
embedding_model = "llama3.1:8b"
model_provider = "ollama"



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

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(docs_parsed)


def create_vector_db() -> Chroma:
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / "my_sources"
    persistent_directory = current_dir / "db" / "antonio_db"

    embeddings = OllamaEmbeddings(model=embedding_model)
    if persistent_directory.exists():
        print("Using existing db")
        return Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("Persistent directory does not exist. Initializing vector store...")
    # Ensure the sources directory exists
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )
    docs = load_documents_from_my_sources(sources_path)
    # Display information about the split documents
    # print("\n--- Document Chunks Information ---")
    # print(f"Number of document chunks: {len(docs)}")
    # print(f"Sample chunk:\n{docs[0].page_content}\n")
    #
    # print("\n--- Creating vector store ---")
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persistent_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def build_prompt_template() -> ChatPromptTemplate:
    messages = [
        (
            "human",
            """
            You are an {user}'s personal assistant.
            You are in charge of question-answering tasks related to their experience with Software engineering and Computer Science.
            Use the following pieces of retrieved context to answer the question. 
            The context is taken from {user}'s CV, Linkedin and other relevant documents.
            If you don't know the answer, just say that you don't know. Use 4 sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """,
        )
    ]
    return ChatPromptTemplate(messages)


def testing_llm() -> None:
    llm = init_chat_model(model, model_provider=model_provider)
    prompt = build_prompt_template()
    db = create_vector_db()

    def retrieve(state: State) -> dict:
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.01},
        )
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                "user": "Antonio Badilla-Olivas",
            }
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": "Should I hire Antonio as a AI Engineer?"})
    print(f"Context: {response['context']}\n\n")
    print(f"Answer: {response['answer']}")


if __name__ == "__main__":
    testing_llm()
