This is a forked version of <a href="https://github.com/Antonio-Tresol/ai_workshop_2025_gen_ai_session.git">ai_workshop_2025_gen_ai_session</a>.

# ai workshop 2025 gen ai session

## Description

This repository contains a comprehensive workshop on building Retrieval-Augmented Generation (RAG) applications using LangChain, Ollama, and ChromaDB.  The project progressively builds a RAG system, starting with basic LLM interactions and culminating in a fully functional, interactive Gradio-based application.  The workshop covers fundamental concepts such as document loading, splitting, vector database creation, prompt engineering, tool integration, and state management using LangGraph.  It also demonstrates structured output generation and error handling.  The final application is a personal assistant chatbot that can answer questions about a fictional person (France Du Pont) based on provided documents (CV, "About Me", and GitHub profile).

## Dependencies

The project relies on the following libraries, as defined in `rag_app/pyproject.toml`:

*   **langchain-ollama** (>=0.2.2,<0.3.0):  Integration with Ollama for local LLM access.
*   **chromadb** (>=0.6.3,<0.7.0): Vector database for storing and retrieving document embeddings.
*   **langchain-community** (>=0.3.16,<0.4.0): Core LangChain components and community integrations.
*   **pypdf** (>=5.2.0,<6.0.0):  For loading and parsing PDF documents.
*   **langchain-chroma** (>=0.2.1,<0.3.0): LangChain integration with ChromaDB.
*   **langgraph** (>=0.2.73,<0.3.0):  For building stateful, multi-actor applications.
*   **langchain-text-splitters** (>=0.3.6,<0.4.0):  For splitting text documents into smaller chunks.
*   **gradio** (>=5.16.0,<6.0.0):  For creating the interactive web UI.

Additionally, the `dev.nix` file specifies the Nix environment, including:

*   `ollama`: For running local LLMs.
*   `python312`:  Python 3.12.
*   `uv`:  For dependency management.
*   `git`: For version control.
*   `gcc-unwrapped`, `pkgs.stdenv.cc.cc`:  C/C++ compilers (likely required by some of the Python packages).

## Installation and Setup

This project uses Nix for environment management and Poetry for Python dependency management.  There are two main ways to run the project: using a Nix-enabled environment in IDX or using Poetry directly.

### 1. Using Uv 

 This approach requires you to manually install system dependencies (like Ollama).

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/EnriqueVilchezL/ai_workshop_2025_gen_ai_session.git
    cd ai_workshop_2025_gen_ai_session/rag_app
    ```

2.  **Install Poetry:**  If you don't have it already, install Poetry:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    More information in <a href="https://docs.astral.sh/uv/getting-started/installation/#pypi">Uv's Documentation </a>

3.  **Install Python Dependencies:**

    ```bash
    uv venv
    uv pip install --compile -r pyproject.toml
    ```

4.  **Install Ollama Manually:**  Download and install Ollama from the official website ([https://ollama.com/](https://ollama.com/)).

5. **Start Ollama (in a separate terminal):**

   ```bash
   ollama serve
   ```

   Keep this terminal running.

6. **Pull an LLM:**

   ```bash
   ollama pull llama3.1
   ```

   so that you can use it for the rag application. You can check other available models in <a href="https://ollama.com/models"> Ollama Models page </a>

7.  **Run the Scripts:**

    You can now run the Python scripts.  Make sure you're in the `rag_app` directory and have the virtual environment created by Uv:

    ```bash
    uv run python 0_llm_basics_with_langchain.py
    ```

### 2. Using IDX

The `dev.nix` describes steps for using Google's IDX. Within IDX, the environment, dependencies and application are automatically run.

## Project Structure

*   **`dev.nix`:**  Defines the Nix development environment, including system dependencies, environment variables, and workspace setup.
*   **`rag_app/`**: Contains the main application code.
    *   **`0_llm_basics_with_langchain.py`**:  Demonstrates basic LLM interaction using LangChain.
    *   **`1_vector_db_basic_simplified.py`**: A simplified introduction to vector databases (ChromaDB) and RAG.
    *   **`1_vector_db_basics.py`**:  A more detailed walkthrough of vector database creation and usage.
    *   **`2_structured_outputs.py`**:  Shows how to generate structured outputs from the LLM.
    *   **`3_deeper_with_langgraph.py`**:  Introduces LangGraph for building more complex, stateful pipelines.
    *   **`bonus_alfred_agent.py`**:  Demonstrates creating a LangChain agent with custom tools.
    *   **`configuration.py`**:  Holds global configuration settings (model names, paths, etc.).
    *   **`error_handler.py`**: Provides a custom error handler with helpful tips for common issues.
    *   **`final_app.py`**: The main Gradio application file.
    *   **`final_graph.py`**:  Defines the final RAG pipeline using LangGraph.
    *   **`final_vector_db.py`**:  Handles the creation and loading of the vector database.
    *   **`pyproject.toml`**:  Defines Python project metadata and dependencies.
    *   **`.gitignore`**: Specifies files and directories to be ignored by Git.
    *   **`dummy_sources/`**: Contains sample text and PDF files used as the knowledge base for the RAG system.  These files describe the fictional person, France Du Pont.
*   **`LICENSE`**:  The MIT License file.

## Running the Application

The final application is in `rag_app/final_app.py`.  To run it:

1.  **Ensure you're in the correct environment**
2.  **Make sure Ollama is running** (`ollama serve` in a separate terminal).
3.  **Navigate to the `rag_app` directory** if you're not already there.
4.  **Run the script:**

    ```bash
    uv run python final_app.py
    ```

This will launch a Gradio web interface in your browser.  You can then interact with the chatbot, asking questions about France Du Pont.

## Key Concepts and Steps

The workshop progressively introduces the following concepts:

1.  **Basic LLM Interaction:**  Using `langchain-ollama` to interact with a local LLM.
2.  **Vector Databases:**  Using ChromaDB to store and retrieve document embeddings.
    *   **Document Loading:** Loading text and PDF files using `TextLoader` and `PyPDFLoader`.
    *   **Document Splitting:**  Breaking documents into smaller chunks using `RecursiveCharacterTextSplitter`.
    *   **Embeddings:** Generating embeddings using `OllamaEmbeddings`.
    *   **Retrieval:**  Retrieving relevant documents based on a query.
3.  **Prompt Engineering:**  Creating effective prompts for the LLM using `ChatPromptTemplate`.
4.  **Structured Output:**  Generating structured output from the LLM using `with_structured_output`.
5.  **LangGraph:** Building stateful pipelines with LangGraph.
    *   **State Management:** Defining the application state using `TypedDict`.
    *   **Nodes and Edges:**  Creating nodes for different processing steps and connecting them with edges.
    *   **Conditional Edges:**  Using `tools_condition` to conditionally route the flow based on LLM output.
    *   **Checkpoints (MemorySaver):** Using a checkpointer to save the graph's state between steps.
6.  **Tools:** Defining and using custom tools with the `@tool` decorator.
7.  **Agents:** Creating a LangChain agent that can use tools to answer questions.
8.  **Gradio UI:** Building an interactive web interface with Gradio.
9. **Error Handling:** The `error_handler.py` contains patterns and tips.

## Contributing

Contributions are welcome! If you find bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.  Copyright by Antonio Badilla-Olivas.

