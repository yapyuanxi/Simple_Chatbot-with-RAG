from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import PyPDF2
import docx

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_word(docx_path):
    """Extract text from a Word file."""
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def load_documents_from_folder(folder_path):
    """Read all PDF and Word files from a folder and extract text."""
    documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_word(file_path)
        else:
            continue  # Skip non-PDF and non-Word files

        documents.append({"filename": filename, "content": text})
    
    return documents

def main(): 

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Define text splitter (for chunking long documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

    # Load documents from a folder
    folder_path = "RAG Information"  # Replace with your folder path
    documents = load_documents_from_folder(folder_path)

    # Process and convert text into embeddings
    all_docs = []
    for doc in documents:
        split_docs = text_splitter.create_documents([doc["content"]])
        all_docs.extend(split_docs)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(all_docs, embedding_model)

    # Save FAISS index
    vectorstore.save_local("faiss_index")

if __name__ =="__main__":
    main()