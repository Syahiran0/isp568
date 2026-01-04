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

def initialize_llm() -> Optional[OpenAI]:
    """Initialize the Ollama LLM client safely."""
    global LLM_CLIENT
    if LLM_CLIENT:
        return LLM_CLIENT
    try:
        print(f"Connecting to Ollama server at: {OLLAMA_BASE_URL}")
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        # Optional test call to verify connection
        try:
            test_response = client.chat.completions.create(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "system", "content": "Test connection"}],
                max_tokens=5
            )
            print("LLM test call successful.")
        except Exception as e:
            print(f"Warning: LLM test call failed: {e}")
        LLM_CLIENT = client
        print(f"LLM client initialized. Target Model: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {e}")
        LLM_CLIENT = None
    return LLM_CLIENT


def generate_llm_response(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """
    Generate a response from Ollama Granite 2B.
    Handles empty responses, format differences, and connection errors.
    """
    client = initialize_llm()
    if client is None:
        return "LLM service not initialized. Ensure Ollama is running and the model is loaded."

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Get the first choice safely
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            # choice.message is an object, use .content
            generated_content = getattr(choice.message, "content", "").strip()
        else:
            generated_content = ""

        if not generated_content:
            print(f"DEBUG: Empty LLM response received: {response}")
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
    # --- 1. Retrieve top relevant context ---
    docs = vector_db.similarity_search(question, k=10)  # increased k for better coverage
    if not docs:
        context_text = "None available."
    else:
        # Limit total length to prevent truncation (approx 3000 tokens)
        context_chunks = []
        total_length = 0
        for doc in docs:
            chunk_len = len(doc.page_content.split())
            if total_length + chunk_len > 3000:
                break
            context_chunks.append(doc.page_content)
            total_length += chunk_len
        context_text = "\n\n".join(context_chunks)

    # --- 2. Include performance info if relevant ---
    performance_context = ""
    keywords = ["performance", "score", "grade", "CLO", "attendance", "test", "assignment"]
    if any(kw in question.lower() for kw in keywords) and student_info:
        performance_context = (
            f"Student Performance: Attendance {student_info['attendance']}%, "
            f"Test {student_info['test_score']}%, "
            f"Assignment {student_info['assignment_score']}%, "
            f"Cognitive Skills {student_info['cognitive']}%, "
            f"Ethics {student_info['ethics']}%. "
            f"Overall Level: {performance_level}."
        )

    # --- 3. Build system prompt with explicit context rule ---
    system_prompt_content = (
        "You are Professor Syahiran, an Academic Mentor. "
        "Answer student academic questions using ONLY the Academic Context provided. "
        "Do not invent or guess. If the answer is not in the context, respond exactly: "
        "'The requested information is not available in the current Academic Context. "
        "Please refer to official university resources.'\n\n"
        f"Academic Context:\n{context_text}\n\n{performance_context}"
    )

    # --- 4. Construct messages ---
    messages = [{"role": "system", "content": system_prompt_content}]
    # include only the last 3 interactions to avoid history overflow
    messages.extend(history[-3:])
    messages.append({"role": "user", "content": question})

    # --- 5. Generate response ---
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
