from openai import OpenAI
from typing import Optional, Dict, List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- IMPORTANT: Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL_NAME = "gemma3:1b"                 # Specify the model you have running in Ollama
CHROMA_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="./my_knowledge_base", embedding_function=embeddings)

LLM_CLIENT: Optional[OpenAI] = None

def initialize_llm():
    """Initializes the OpenAI client pointing to the local Ollama server."""
    global LLM_CLIENT
    if LLM_CLIENT is None:
        try:
            print(f"Attempting to connect to Ollama server at: {OLLAMA_BASE_URL}")
            LLM_CLIENT = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama" 
            )
            print(f"LLM client initialized. Target Model: {OLLAMA_MODEL_NAME}")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client for Ollama. Error: {e}")
            LLM_CLIENT = None
            
    return LLM_CLIENT

def generate_llm_response(messages: List[Dict]) -> str:
    """
    Generates a response from the LLM using the chat completion interface based on a list of messages.
    """
    client = initialize_llm()
    if client is None:
        return "LLM service is not initialized. Please ensure Ollama is running and the model is loaded."

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=messages, # Pass the entire message list
            temperature=0.7,
            max_tokens=1024,
        )
        
        generated_content = response.choices[0].message.content.strip()
        
        if not generated_content:
            return "The language model failed to generate a suggestion. This can happen if the model's output is blank or too short."
        
        return generated_content
        
    except Exception as e:
        print(f"LLM generation error: {e}")
        return f"An error occurred while communicating with the local Ollama server. Check if the model '{OLLAMA_MODEL_NAME}' is running: {e}"

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
        "You are an Academic Mentor focused on holistic student development. Review the student's evaluation, paying close attention to how their ethics and cognitive skills influence their overall performance score. Write one concise and encouraging paragraph that explains how improving their weakest metric will boost their overall academic standing. Use professional, plain text only. Avoid lists, headers, or markdown formatting."
        )
        user_prompt = context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return generate_llm_response(messages)
    except KeyError as e:
        return f"Error: Missing input key {e}"


def get_lecturer_chat_response(performance_level: str, question: str, history: List[Dict]) -> str:
    # 1. Try to get context
    try:
        docs = vector_db.similarity_search(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception:
        # If the database is empty or missing, provide empty context
        context_text = "No specific academic documents are currently available."

    # 2. Build the prompt
    # If context_text is empty, the Professor will rely on his 'Lecturer' persona
    system_prompt_content = (
        f"You are Professor Syahiran, a knowledgeable and strict lecturer. "
        f"Student Performance: {performance_level}. "
        f"Academic Context: {context_text if context_text else 'None available.'} "
        f"Answer the question concisely in 3-4 sentences. "
        f"If no context is provided, answer based on general academic best practices."
    )
    
    messages = [{"role": "system", "content": system_prompt_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": question}) 
    
    return generate_llm_response(messages)

def process_documents(file_path: str):
    """Parses PDF/Word and splits into chunks for the database."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        return []

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(docs)