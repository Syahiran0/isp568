from openai import OpenAI
from typing import Optional, Dict, List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "granite3-dense:2b"  # Updated model for fast RAG responses
CHROMA_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="./my_knowledge_base", embedding_function=embeddings)

LLM_CLIENT: Optional[OpenAI] = None

def initialize_llm():
    """Initializes the Ollama client for Granite 2B."""
    global LLM_CLIENT
    if LLM_CLIENT is None:
        try:
            print(f"Connecting to Ollama server at: {OLLAMA_BASE_URL}")
            LLM_CLIENT = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama"
            )
            print(f"LLM client initialized. Target Model: {OLLAMA_MODEL_NAME}")
        except Exception as e:
            print(f"ERROR: Failed to initialize LLM client: {e}")
            LLM_CLIENT = None
    return LLM_CLIENT

def generate_llm_response(messages: List[Dict]) -> str:
    """Generates a response from Granite 2B using the chat completion API."""
    client = initialize_llm()
    if client is None:
        return "LLM service not initialized. Ensure Ollama is running and the model is loaded."
    
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=messages,
            temperature=0.2,   # Lower temperature for factual output
            max_tokens=1024
        )
        generated_content = response.choices[0].message.content.strip()
        if not generated_content:
            return "The language model returned an empty response."
        return generated_content
    except Exception as e:
        print(f"LLM generation error: {e}")
        return f"Error communicating with Ollama '{OLLAMA_MODEL_NAME}': {e}"

def get_ai_suggestion(inputs: Dict, level: str, score: float) -> str:
    try:
        context = (
            f"The student received the following evaluation scores: "
            f"Attendance: {inputs['attendance']}%, "
            f"Test Score: {inputs['test_score']}%, "
            f"Assignment Score: {inputs['assignment_score']}%, "
            f"Ethics: {inputs['ethics']}%, "
            f"Cognitive Skills: {inputs['cognitive']}%. "
            f"The calculated overall performance level is {level} "
            f"(Fuzzy Score: {score}/100)."
        )
        system_prompt = (
            "You are an Academic Mentor focused on holistic student development. "
            "Review the student's evaluation, paying attention to ethics and cognitive skills. "
            "Write one concise, encouraging paragraph explaining how improving the weakest metric can boost overall performance. "
            "Use professional plain text only. No lists, headers, or markdown formatting."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        return generate_llm_response(messages)
    except KeyError as e:
        return f"Error: Missing input key {e}"

def get_lecturer_chat_response(performance_level: str, question: str, history: List[Dict]) -> str:
    # Retrieve context from vector DB
    try:
        docs = vector_db.similarity_search(question, k=5)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception:
        context_text = ""

    # Strict RAG system prompt
    system_prompt_content = (
        "You are Professor Syahiran responding to student academic questions. "
        "The Academic Context provided below is official and authoritative. "
        "Rules: Use only the Academic Context for factual or administrative questions. "
        "If a list, number, or fact is present, reproduce it exactly. "
        "Do not add, remove, rephrase, or invent any information. "
        "If information is missing, clearly state that it is not provided. "
        "Respond in 1–3 sentences, plain text only. No lists, markdown, or roleplay. "
        f"Academic Context:\n{context_text if context_text else 'None available.'}"
    )

    messages = [{"role": "system", "content": system_prompt_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return generate_llm_response(messages)

def process_documents(file_path: str):
    """Load and split PDF or Word documents into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        return []

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)
