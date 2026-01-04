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

def get_lecturer_chat_response(performance_level: str, question: str, history: List[Dict], student_info: Optional[Dict] = None) -> str:
    # 1. Retrieve context from vector DB
    docs = vector_db.similarity_search(question, k=5)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Include performance info only if question is related
    performance_context = ""
    keywords = ["performance", "score", "grade", "CLO", "attendance", "test", "assignment"]
    if any(kw in question.lower() for kw in keywords) and student_info:
        performance_context = (
            f"Student Performance: Attendance {student_info['attendance']}%, "
            f"Test {student_info['test_score']}%, "
            f"Assignment {student_info['assignment_score']}%, "
            f"Cognitive Skills {student_info['cognitive']}%, "
            f"Ethics {student_info['ethics']}%. "
            f"Overall Level: {performance_level}"
        )

    # 3. Build system prompt
    system_prompt_content = (
        "You are Professor Syahiran responding to student academic questions. "
        "Rules:\n"
        "- Include student performance details only if relevant to the question.\n"
        "- Use only the Academic Context for factual information.\n"
        "- Do not invent or hallucinate.\n"
        "- If information is missing, respond: "
        "'The requested information is not available in the current Academic Context. "
        "Please refer to official university resources.'\n"
        f"Academic Context:\n{context_text if context_text else 'None available.'}\n"
        f"{performance_context}"
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
