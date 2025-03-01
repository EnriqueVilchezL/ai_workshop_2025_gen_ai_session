"""
Workshop Guide: Building a Retrieval-Augmented Generation (RAG) Pipeline – Final Step

This file demonstrates an advanced RAG system that:
1. Retrieves relevant document chunks from a vector database.
2. Validates the retrieved documents using an LLM, with a retry mechanism if needed.
3. Generates an answer using an LLM and the retrieved context.
4. Assembles the entire process into a state graph defining the pipeline.

Key concepts include:
    - The use of a retrieval tool that leverages document validation.
    - Binding tools (like retrieval) with an LLM.
    - Sequencing the process steps using a state graph.
    - Detailed logging and retry logic to ensure document relevance.
"""

# -------------------------------
# Import necessary modules and libraries.
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from final_vector_db import create_vector_db
from configuration import model, model_provider, user, db_search_kwargs, db_search_type


# Initialize the language model (LLM) and the vector database.
llm = init_chat_model(model, model_provider=model_provider)
db = create_vector_db()


# -------------------------------
# TOOL: retrieve
# This function is decorated as a tool so it can be dynamically bound in the pipeline.
@tool(response_format="content_and_artifact")
def retrieve(query: str) -> tuple:
    """
    Retrieve information related to a query with document validation.

    This function performs the following steps:
      1. Uses the vector database's retriever to fetch a set of documents relevant to the query.
      2. Summarizes each retrieved document (first 300 characters) for validation.
      3. Prompts the LLM to validate if the retrieved documents are useful.
      4. Retries retrieval (up to 3 times) if the LLM indicates the documents are not sufficiently relevant.

    Args:
        query (str): The user's query.

    Returns:
        tuple: A tuple containing a serialized string (with document details) and the list of retrieved documents.
    """
    max_attempts = 2
    attempt = 0
    while attempt < max_attempts:
        # Retrieve documents from the vector database using a similarity search.
        retriever = db.as_retriever(
            search_type=db_search_type,
            search_kwargs=db_search_kwargs,
        )
        retrieved_docs = retriever.invoke(query)

        # Create a summary of each document for the LLM evaluation.
        docs_summary = "\n\n".join(
            f"Document {i + 1} (Source: {doc.metadata.get('source', 'unknown')}): "
            f"{doc.page_content[:300]}..."  # Display only the first 300 characters.
            for i, doc in enumerate(retrieved_docs)
        )

        # Create a validation prompt for the LLM to decide on document relevance.
        validation_prompt = f"""
            You are an expert evaluator. Given the query: "{query}"
            and the following document excerpts:
            {docs_summary}
            
            Determine if these documents are useful for answering the query.
            If they are highly relevant, reply with "OK". Otherwise, reply with "Retry".
            """
        # Invoke the language model to validate the retrieved documents.
        validation_response = llm.invoke([SystemMessage(validation_prompt)])

        # Log the validation result for this attempt.
        print(
            f"Validation attempt {attempt + 1}: {validation_response.content.strip()}"
        )

        # If the LLM confirms the documents are relevant, serialize and return them.
        if "ok" in validation_response.content.lower():
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        # If not, attempt retrieval again.
        attempt += 1
        print("LLM requested a retry to find better documents. Retrying retrieval...")

    # After reaching the maximum attempts, return the last retrieved documents.
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState) -> dict:
    """
    Generate a tool call for retrieval or a direct response.

    The function binds the LLM with the retrieval tool and uses the conversation state
    ("messages") to generate an appropriate response.

    Args:
        state (MessagesState): The conversation state containing messages exchanged so far.

    Returns:
        dict: A dictionary with the generated response as a list of messages.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """
    Generate the final answer based on retrieved context.

    Steps:
      1. Extracts tool (retrieval) messages from the conversation history.
      2. Combines these messages into a single document context.
      3. Constructs a formatted system prompt message that includes instructions and the retrieved context.
      4. Prepares the final conversation messages by combining system instructions with non-tool messages.
      5. Invokes the LLM to generate and return the answer.

    Args:
        state (MessagesState): The conversation state containing messages.

    Returns:
        dict: A dictionary containing the final answer as a list of messages.
    """
    # Collect tool messages (responses from the retrieval tool) from the conversation.
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Concatenate content from each tool message to form the document context.
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Build the system message that instructs the LLM on how to generate an answer.
    system_message_content = (
        f"""
        You are {user}'s personal assistant, your name is Alfred 🧐.
        You are in charge of question-answering tasks related to their experience with Software Engineering and Computer Science.
        Use the following pieces of retrieved context to answer the question.
        If asked, you can answer questions about their life, interests, and hobbies if sources allow.
        The context is taken from {user}'s CV, LinkedIn, and other relevant documents.
        If you don't know the answer, just say that you don't know. Your statements must be supported by the sources.
        Use no more than 5 sentences and be friendly.
        """
        "\n\n"
        f"{docs_content}"
    )

    # Filter out messages that are tool-generated and combine remaining conversation history.
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    # Prepend the system message to create the final prompt.
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Invoke the language model with the final prompt.
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_rag_graph() -> tuple:
    """
    Construct the RAG pipeline state graph.

    The function performs the following:
      1. Adds nodes for processing the conversation via query_or_respond, retrieval, and final answer generation.
      2. Specifies the entry point for the graph.
      3. Adds conditional edges to decide when to invoke the tools.
      4. Connects nodes to ensure correct sequential processing.
      5. Attaches memory saving for checkpointing purposes.

    Returns:
        tuple: A tuple containing the compiled graph and a configuration dictionary.
    """
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([retrieve]))
    graph_builder.add_node(generate)

    # Set the starting node.
    graph_builder.set_entry_point("query_or_respond")
    # Configure conditional edges: if the output of query_or_respond suggests use of tools, go to "tools".
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    # After tool retrieval, move to answer generation.
    graph_builder.add_edge("tools", "generate")
    # End the graph after generation.
    graph_builder.add_edge("generate", END)

    # Use memory saver to checkpoint graph state.
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Additional configuration, the memory saver need this thread id to work
    config = {"configurable": {"thread_id": "abc123"}}
    return graph, config
