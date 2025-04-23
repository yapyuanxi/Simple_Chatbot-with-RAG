# Simple Chatbot with Retrieval-Augmented Generation (RAG)

This repository provides an implementation of a conversational AI chatbot leveraging large language models integrated with Retrieval-Augmented Generation (RAG) using a FAISS vector store and multilingual embeddings. The chatbot is presented through a user-friendly Streamlit interface, designed for responsive and context-aware interactions.

## Features

* **Flexible Language Models:** Supports various large language models for text generation.
* **Retrieval-Augmented Generation (RAG):** Enhances responses using contextual retrieval from a FAISS vector database.
* **Multilingual Embeddings:** Implements multilingual embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) for robust document representation.
* **Interactive Interface:** Utilizes Streamlit to create an intuitive and interactive chatbot interface.
* **Support for PDF and Word Files:** Automatically extracts and processes content from PDF and Word documents for creating the knowledge base.

## Project Structure

```bash
Simple_Chatbot-with-RAG/
├── backend.py              # Backend logic for model loading, inference, and retrieval
├── frontend.py             # Streamlit frontend application
├── preprocess_documents.py # Document preprocessing and FAISS index creation
├── faiss_index/            # Directory storing FAISS index
├── RAG Information/        # Folder containing documents (PDF and Word) to build knowledge base
└── README.md               # This documentation
```
## Setup Instructions

1.  **Clone Repository**
    ```bash
    git clone https://github.com/yapyuanxi/Simple_Chatbot-with-RAG.git
    cd Simple_Chatbot-with-RAG
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch transformers langchain faiss-cpu sentence-transformers streamlit pypdf2 python-docx
    ```
    *Note: Use `faiss-gpu` if CUDA is available.*

3.  **Prepare Knowledge Base**
    * Place your PDF and Word documents into the `RAG Information` folder.
    * Run the preprocessing script:
        ```bash
        python RAG_Extraction.py
        ```
        This will create embeddings and store them as a FAISS index.

4.  **Download and Setup Language Model (e.g., DeepSeek)**
    To use a specific language model such as DeepSeek, you can download it from Hugging Face. Here's an example:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ```

5.  **Run Chatbot Interface**
    Launch the Streamlit frontend:
    ```bash
    streamlit run frontend.py
    ```
    Navigate to the provided local URL to interact with the chatbot.

## Usage

* **Chat Interaction:** Type queries directly into the chat interface to receive contextually informed responses.
* **Debug Information:** Enable the "Show debug info" option in the Streamlit sidebar to view intermediate outputs and debugging details.

## Technical Details

* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Vectorstore:** FAISS
* **Frontend:** Streamlit

## Acknowledgements

* HuggingFace Transformers
* LangChain
* FAISS
* Streamlit

## Author

Yap Yuan Xi - [https://github.com/yapyuanxi](https://github.com/yapyuanxi)
