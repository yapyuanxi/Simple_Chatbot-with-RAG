# backend.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import autocast
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model onto GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
model.to(device)

# Embeddings & Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

def deepseek_generate(prompt: str) -> str:
    """Generate text given a prompt using the loaded model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with autocast(device_type=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
        output = model.generate(
            **inputs,
            max_new_tokens=1000
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def retrieve_context(query: str) -> str:
    """Retrieve the most relevant chunks for a given query from FAISS."""
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in retrieved_docs])

def deepseek_rag_pipeline(query: str) -> str:
    """Construct a prompt with retrieved knowledge and pass to `deepseek_generate`."""
    retrieved_context = retrieve_context(query)

    # Construct prompt with retrieved knowledge
    full_prompt = f"""
    You are an AI assistant. Use the retrieved knowledge below to answer accurately:

    Retrieved Context:
    {retrieved_context}

    Question: {query}
    Answer:
    """

    # Generate response
    response = deepseek_generate(full_prompt)
    
    return response
