from fastapi import FastAPI
from api_router import router
from llm_service import initialize_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Fuzzy Student Evaluation System",
    description="An API that evaluates student performance using fuzzy logic and provides AI-generated academic suggestions.",
    version="1.1.1"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initializes the LLM on application startup."""
    print("--- Starting up application and initializing LLM client... ---")
    initialize_llm()
    print("--- Application ready. ---")

origins = [
    "http://localhost:5173",   # React
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
# uvicorn main:app --reload