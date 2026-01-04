import os
from fastapi import APIRouter, HTTPException, Depends,  UploadFile, File
from ollama import embeddings
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import shutil
import time
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"


# --- Module Imports ---
try:
    from fuzzy_system import evaluate_score, get_performance_level 
    from llm_service import initialize_llm, get_ai_suggestion, get_lecturer_chat_response  , process_documents, vector_db
    from reports import generate_student_report_pdf
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: Could not find supporting modules. Error: {e}")
    raise e

router = APIRouter(prefix="/api/v1")

# --- Pydantic Models ---

class EvaluationInput(BaseModel):
    """Input model for student scores."""
    attendance: float = Field(..., ge=0, le=100, description="Attendance (0-100)")
    test_score: float = Field(..., ge=0, le=100, description="Test Score (0-100)")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment Score (0-100)")
    ethics: float = Field(..., ge=0, le=100, description="Professionalism/Ethics (0-100)")
    cognitive: float = Field(..., ge=0, le=100, description="Cognitive Ability (0-100)")

class Message(BaseModel):
    """Single message in the conversation history."""
    role: str = Field(..., description="The role of the message sender ('user' or 'assistant').")
    content: str = Field(..., description="The content of the message.")

class ChatQuery(BaseModel):
    """Input model for the chat lecturer endpoint, now including history."""
    student_performance_level: str = Field(..., description="The calculated performance level (e.g., 'Weak', 'Average', 'Excellent').")
    # History must be passed by the client
    history: List[Message] = Field(default=[], description="The previous messages in the conversation (user and assistant).")
    # FIX: Changed field name from 'current_question' to 'question' to match client's payload
    question: str = Field(..., description="The student's new question for the lecturer.")


# Dependency to check LLM status
def check_llm_status():
    if not initialize_llm():
        raise HTTPException(status_code=503, detail="LLM service is unavailable. Ensure Ollama is running and the model is loaded.")

# --- Endpoint Implementations ---

# --- /evaluate endpoint ---
@router.post("/evaluate", response_model=Dict[str, Any], tags=["Evaluation"])
async def evaluate_student(input_data: EvaluationInput):
    """
    Evaluate the student using fuzzy logic and return score, performance level, and triggered rules.
    """
    try:
        data_dict = input_data.model_dump()
        result = evaluate_score(data_dict)

        return {
            "status": "success",
            "inputs": data_dict,  # Keep original inputs for suggestion
            "fuzzy_score": result['fuzzy_score'],
            "performance_level": result['performance_level'],
            "rules_triggered": result['applied_rules']  # This contains IF...THEN rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- /suggestion endpoint ---
@router.post("/suggestion", response_model=Dict[str, Any], dependencies=[Depends(check_llm_status)], tags=["LLM Advisor"])
async def get_ai_suggestion_endpoint(input_data: EvaluationInput):
    """
    Generates a personalized academic suggestion based on the student's fuzzy evaluation scores.
    Uses a local LLM (Ollama) for generating suggestions.
    """
    try:
        # 1. Directly use evaluate_score instead of calling the async endpoint
        data_dict = input_data.model_dump()
        evaluation_result = evaluate_score(data_dict)  # Returns dict with applied_rules and fuzzy_score

        inputs = evaluation_result['inputs']
        level = evaluation_result['performance_level']
        score = evaluation_result['fuzzy_score']
        active_rules = evaluation_result.get('applied_rules', [])

        # 2. Generate AI suggestion in a threadpool to avoid blocking
        suggestion = await run_in_threadpool(
            get_ai_suggestion,
            inputs,
            level,
            score
        )

        # 3. Check for LLM errors
        if suggestion.startswith("LLM service is not initialized") or suggestion.startswith("An error occurred"):
            raise HTTPException(status_code=500, detail=suggestion)

        return {
            "status": "success",
            "fuzzy_score": score,
            "performance_level": level,
            "applied_rules": active_rules,  # includes IF…THEN logic
            "suggestion": suggestion
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestion: {str(e)}")

    
@router.post("/chat", response_model=Dict[str, Any], dependencies=[Depends(check_llm_status)], tags=["LLM Lecturer"])
async def chat_with_lecturer_endpoint(chat_data: ChatQuery):
    """
    Acts as a lecturer (using Ollama) who answers questions related to student evaluation and suggestions, 
    maintaining conversation history.
    """
    # Prepare history for the service function
    history_messages = [msg.model_dump() for msg in chat_data.history]
    
    # FIX: Use chat_data.question (the new field name from the Pydantic model)
    current_question = chat_data.question 

    # Run synchronous LLM chat call in a threadpool
    answer = await run_in_threadpool(
        get_lecturer_chat_response,
        chat_data.student_performance_level,
        current_question, # Pass the question 
        history_messages # Pass the history list
    )

    if answer.startswith("LLM service is not initialized") or answer.startswith("An error occurred"):
        raise HTTPException(status_code=500, detail=answer)
    
    return {
        "status": "success",
        "answer": answer
    }

@router.post("/upload-knowledge", tags=["LLM Lecturer"])
async def upload_knowledge_document(file: UploadFile = File(...)):
    """
    Uploads a PDF or Word doc, splits it into chunks, 
    and adds it to the ChromaDB vector store.
    """
    # 1. Ensure a temporary directory exists
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        # 2. Save the uploaded file to disk temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Process the document into chunks
        # This calls the function we created in the previous step
        chunks = process_documents(file_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="Unsupported file format or empty file.")

        # 4. Add to the Vector Database
        # This creates the mathematical embeddings and saves them to ./chroma_db
        vector_db.add_documents(chunks)

        return {
            "status": "success", 
            "message": f"Document '{file.filename}' processed and added to memory.",
            "chunks_added": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
    finally:
        # 5. Clean up: Delete the temp file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    
@router.get("/report/download", tags=["Reports"])
async def download_pdf_report(
    attendance: float, 
    test_score: float, 
    assignment_score: float,
    ethics: float = 0.0,
    cognitive: float = 0.0
):
    """
    Generates a PDF report for a student's evaluation, including the AI suggestion, and returns it for download.
    """
    try:
        # 1. Create input dictionary for fuzzy evaluation
        input_data_dict = {
            "attendance": attendance,
            "test_score": test_score,
            "assignment_score": assignment_score,
            "ethics": ethics,
            "cognitive": cognitive
        }

        # 2. Evaluate student using fuzzy logic
        evaluation_result = evaluate_score(input_data_dict)  # returns dict with applied_rules, fuzzy_score, etc.

        # 3. Generate AI suggestion (run in threadpool to avoid blocking)
        suggestion_text = await run_in_threadpool(
            get_ai_suggestion,
            evaluation_result['inputs'],
            evaluation_result['performance_level'],
            evaluation_result['fuzzy_score']
        )

        # 4. Generate PDF report (also in threadpool)
        pdf_buffer = await run_in_threadpool(
            generate_student_report_pdf,
            evaluation_result['inputs'],  # original input scores
            evaluation_result,           # evaluation results including applied_rules
            suggestion_text              # AI suggestion text
        )

        # 5. Return PDF as a downloadable file
        report_filename = f"student_report_{evaluation_result['performance_level']}_{int(time.time())}.pdf"

        return Response(
            content=pdf_buffer.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={report_filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")

@router.post("/reset-memory", tags=["LLM Lecturer"])
async def reset_all_knowledge():
    global vector_db 
    
    try:
        # 1. Delete the collection
        vector_db.delete_collection()
        print("Collection deleted.")
    except Exception as e:
        print(f"Collection already empty or error: {e}")
    
    # 2. MANDATORY: Re-create the collection so the variable is valid again
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    return {"status": "success", "message": "Memory wiped. Professor is ready for new files or general questions."}