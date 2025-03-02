# documentos check
# parsearlos check
# meterlos en la base de datos vectorial
from configuration import (
    sources_dir,
    db_dir,
    embedding_model,
    db_search_type,
    db_search_kwargs,
    user,
    model,
    model_provider,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from error_handler import handle_exception


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / sources_dir

    source_documents = []
    for file in sources_path.iterdir():
        if file.suffix in {".pdf", ".txt"}:
            source_documents.append({"file": file, "extension": file.suffix})

    # for source in source_documents:
    #     print(source["file"].name)

    parsed_document = []
    for source in source_documents:
        file_path = source["file"]
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
        parsed_document.extend(documents)

    # for source in parsed_document:
    #     print("------------------------------------")
    #     print(source)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    splitted_documents = text_splitter.split_documents(parsed_document)

    # for split in splitted_documents:
    #     print("------------------------------------")
    #     print(split)

    persistent_directory = current_dir / db_dir
    embedder = OllamaEmbeddings(model=embedding_model)
    collection_meta = {"hnsw:space": "cosine"}

    if persistent_directory.exists():
        db = Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embedder,
            collection_metadata=collection_meta,
        )
    else:
        db = Chroma.from_documents(
            documents=splitted_documents,
            embedding=embedder,
            persist_directory=str(persistent_directory),
            collection_metadata=collection_meta,
        )

    retriever = db.as_retriever(
        search_type=db_search_type, search_kwargs=db_search_kwargs
    )
    # retrieved_document = retriever.invoke(f"should we hire {user} as an AI Engineer?")
    # for doc in retrieved_document:
    #     print("------------------------------------")
    #     print(doc.page_content)

    llm = init_chat_model(model=model, model_provider=model_provider)

    messages = [
        (
            "human",
            """You are a helpful and insightfull butler name Alfred working for {user}. Keep your answer short.
        The only thing you know about our current state is the conversation history, do no assume anything.
        If conversion history is empty, we are starting a new conversation and you should remember what we talk about.
        Do not invent anything that is not inside our conversation or you knowledge of the world.
        - conversion history messages (your answer have the word 'model' before them): {history} 
        - last message: {message}
        - relevant documents to answer query (ignore if empty): {documents}
        """,
        )
    ]

    prompt = ChatPromptTemplate(messages)

    # response = llm.invoke(prompt.invoke({"user": "enrique"}))

    # print(f"model answer: {response.content}")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive chat.")
            break
        relevant_documents = retriever.invoke(user_input)
        relevant_docs_content = "\n\n".join(
            doc.page_content for doc in relevant_documents
        )
        retrieved_document = retriever.invoke(f"should we hire {user} as an AI Engineer?")

        print(f"-------------- RETRIEVED DOCUMENT FOR QUERY '{user_input}'----------------------")
        for doc in retrieved_document:
            print("------------------------------------")
            print(doc.page_content)
            print("------------------------------------")
        print()

        formated_message = prompt.invoke(
            {
                "user": user,
                "history": "\n".join(history),
                "message": user_input,
                "documents": relevant_docs_content,
            }
        )
        # response = llm.invoke(formated_message)
        print("model answer: ", end="")
        # streaming the answer
        response = ""
        for chunk in llm.stream(formated_message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        print()
        history.append(f"user: {user_input}")
        history.append(f"model: {response}")


if __name__ == "__main__":
    # Run the pipeline test when this script is executed.
    try:
        main()
    except Exception as e:
        handle_exception(e, model)
