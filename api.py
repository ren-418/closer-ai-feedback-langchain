from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from langchain_script.evaluator import SalesCallEvaluator
from embeddings.pinecone_store import PineconeManager
import json

app = FastAPI(
    title="AI Sales Call Evaluator API",
    description="API for evaluating sales calls using RAG and LLM analysis with reference file tracking",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our evaluator
evaluator = SalesCallEvaluator()
pinecone_manager = PineconeManager()

class TranscriptRequest(BaseModel):
    transcript: str
    top_k: Optional[int] = 3

class TranscriptResponse(BaseModel):
    transcript_analysis: Dict
    chunk_analyses: List[Dict]
    final_analysis: Dict
    metadata: Dict
    status: str

@app.post("/evaluate", response_model=TranscriptResponse)
async def evaluate_transcript(request: TranscriptRequest):
    """
    Evaluate a sales call transcript and return detailed analysis with reference file tracking.
    """
    try:
        result = evaluator.evaluate_transcript(
            request.transcript,
            top_k=request.top_k
        )
        if result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=result.get('error', 'Evaluation failed'))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/file")
async def evaluate_transcript_file(file: UploadFile = File(...), top_k: int = 3):
    """
    Evaluate a sales call transcript from an uploaded file.
    """
    try:
        content = await file.read()
        transcript = content.decode('utf-8')
        result = evaluator.evaluate_transcript(transcript, top_k=top_k)
        if result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=result.get('error', 'Evaluation failed'))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the Pinecone index and evaluation system.
    """
    try:
        stats = pinecone_manager.get_index_stats()
        return {
            "pinecone_stats": stats,
            "status": "healthy",
            "system_info": {
                "evaluator_version": "1.0.0",
                "analysis_features": [
                    "Reference file tracking",
                    "Enhanced chunk analysis", 
                    "Professional scoring",
                    "Coaching recommendations"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 