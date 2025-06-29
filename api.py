from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import hashlib
import jwt
from datetime import datetime, timedelta
from langchain_script.evaluator import SalesCallEvaluator
from embeddings.pinecone_store import PineconeManager
from database.database_manager import db_manager
import json

app = FastAPI(
    title="AI Sales Call Evaluator API",
    description="Professional API for evaluating sales calls with business management features",
    version="2.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

# Initialize components
evaluator = SalesCallEvaluator()
pinecone_manager = PineconeManager()

# Pydantic models
class UserLogin(BaseModel):
    email: str
    password: str

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class CloserCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    hire_date: Optional[str] = None

class CallUpload(BaseModel):
    closer_name: str
    filename: Optional[str] = None
    call_date: Optional[str] = None

class CallUpdate(BaseModel):
    status: str

class TranscriptRequest(BaseModel):
    transcript: str
    closer_name: str
    filename: Optional[str] = None
    call_date: Optional[str] = None
    top_k: Optional[int] = 3

class TranscriptResponse(BaseModel):
    call_id: str
    transcript_analysis: Dict
    chunk_analyses: List[Dict]
    final_analysis: Dict
    metadata: Dict
    status: str

# Authentication functions
def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token and return user data."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user = db_manager.get_user_by_email(payload.get("email"))
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication endpoints
@app.post("/auth/register")
async def register_user(user: UserCreate):
    """Register a new admin user."""
    try:
        # Check if user already exists
        existing_user = db_manager.get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user
        password_hash = hash_password(user.password)
        new_user = db_manager.create_user(user.email, password_hash, user.full_name)
        
        if not new_user:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return {"message": "User created successfully", "user_id": new_user['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login_user(user: UserLogin):
    """Login user and return access token."""
    try:
        # Get user from database
        db_user = db_manager.get_user_by_email(user.email)
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        password_hash = hash_password(user.password)
        if db_user['password_hash'] != password_hash:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create access token
        access_token = create_access_token({"email": user.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": db_user['id'],
                "email": db_user['email'],
                "full_name": db_user['full_name'],
                "role": db_user['role']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Closer management endpoints
@app.post("/closers", dependencies=[Depends(verify_token)])
async def create_closer(closer: CloserCreate):
    """Create a new closer."""
    try:
        new_closer = db_manager.create_closer(
            closer.name, 
            closer.email, 
            closer.phone, 
            closer.hire_date
        )
        if not new_closer:
            raise HTTPException(status_code=500, detail="Failed to create closer")
        return new_closer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/closers", dependencies=[Depends(verify_token)])
async def get_closers():
    """Get all closers."""
    try:
        closers = db_manager.get_all_closers()
        return {"closers": closers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/closers/{closer_id}", dependencies=[Depends(verify_token)])
async def get_closer(closer_id: str):
    """Get specific closer by ID."""
    try:
        closer = db_manager.get_closer_by_id(closer_id)
        if not closer:
            raise HTTPException(status_code=404, detail="Closer not found")
        return closer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Call management endpoints
@app.post("/calls/upload", dependencies=[Depends(verify_token)])
async def upload_call_file(
    file: UploadFile = File(...),
    closer_name: str = None,
    call_date: str = None
):
    """Upload and analyze a call transcript file."""
    try:
        if not closer_name:
            raise HTTPException(status_code=400, detail="closer_name is required")
        
        # Read file content
        content = await file.read()
        transcript = content.decode('utf-8')
        
        # Create call record
        call_record = db_manager.create_call(
            closer_name=closer_name,
            transcript_text=transcript,
            filename=file.filename,
            call_date=call_date
        )
        
        if not call_record:
            raise HTTPException(status_code=500, detail="Failed to create call record")
        
        # Analyze the transcript
        analysis_result = evaluator.evaluate_transcript(transcript)
        
        if analysis_result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=analysis_result.get('error', 'Analysis failed'))
        
        # Store analysis results
        success = db_manager.update_call_analysis(call_record['id'], analysis_result)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store analysis results")
        
        return {
            "call_id": call_record['id'],
            "message": "Call uploaded and analyzed successfully",
            "analysis": analysis_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calls/evaluate", dependencies=[Depends(verify_token)])
async def evaluate_transcript(request: TranscriptRequest):
    """Evaluate a transcript and store results."""
    try:
        # Create call record
        call_record = db_manager.create_call(
            closer_name=request.closer_name,
            transcript_text=request.transcript,
            filename=request.filename,
            call_date=request.call_date
        )
        
        if not call_record:
            raise HTTPException(status_code=500, detail="Failed to create call record")
        
        # Analyze the transcript
        analysis_result = evaluator.evaluate_transcript(
            request.transcript,
            top_k=request.top_k
        )
        
        if analysis_result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=analysis_result.get('error', 'Analysis failed'))
        
        # Store analysis results
        success = db_manager.update_call_analysis(call_record['id'], analysis_result)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store analysis results")
        
        return {
            "call_id": call_record['id'],
            "transcript_analysis": analysis_result.get('transcript_analysis', {}),
            "chunk_analyses": analysis_result.get('chunk_analyses', []),
            "final_analysis": analysis_result.get('final_analysis', {}),
            "metadata": analysis_result.get('metadata', {}),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calls", dependencies=[Depends(verify_token)])
async def get_calls(
    closer_name: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    """Get calls with optional filtering."""
    try:
        calls = db_manager.get_calls(closer_name, status, start_date, end_date, limit)
        return {"calls": calls, "total": len(calls)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calls/{call_id}", dependencies=[Depends(verify_token)])
async def get_call(call_id: str):
    """Get specific call with full analysis."""
    try:
        call = db_manager.get_call_by_id(call_id)
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        return call
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/calls/{call_id}/status", dependencies=[Depends(verify_token)])
async def update_call_status(call_id: str, call_update: CallUpdate):
    """Update call status."""
    try:
        success = db_manager.update_call_status(call_id, call_update.status)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update call status")
        return {"message": "Call status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/closer/{closer_name}", dependencies=[Depends(verify_token)])
async def get_closer_analytics(closer_name: str, days: int = 30):
    """Get performance analytics for a specific closer."""
    try:
        analytics = db_manager.get_closer_performance(closer_name, days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/team", dependencies=[Depends(verify_token)])
async def get_team_analytics():
    """Get team-wide analytics and leaderboard."""
    try:
        analytics = db_manager.get_team_analytics()
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/leaderboard", dependencies=[Depends(verify_token)])
async def get_leaderboard():
    """Get closers leaderboard."""
    try:
        analytics = db_manager.get_team_analytics()
        return {
            "leaderboard": analytics.get('leaderboard', []),
            "top_performers": analytics.get('top_performers', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.get("/search", dependencies=[Depends(verify_token)])
async def search_calls(search_term: str, limit: int = 50):
    """Search calls by closer name or filename."""
    try:
        results = db_manager.search_calls(search_term, limit)
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints (for backward compatibility)
@app.post("/evaluate", response_model=TranscriptResponse)
async def evaluate_transcript_legacy(request: TranscriptRequest):
    """Legacy endpoint for transcript evaluation (no database storage)."""
    try:
        result = evaluator.evaluate_transcript(
            request.transcript,
            top_k=request.top_k
        )
        if result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=result.get('error', 'Evaluation failed'))
        
        # Add dummy call_id for compatibility
        result['call_id'] = 'legacy-' + datetime.now().isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = pinecone_manager.get_index_stats()
        return {
            "pinecone_stats": stats,
            "status": "healthy",
            "system_info": {
                "evaluator_version": "2.0.0",
                "analysis_features": [
                    "Reference file tracking",
                    "Enhanced chunk analysis", 
                    "Professional scoring",
                    "Coaching recommendations",
                    "Business management features"
                ],
                "database_enabled": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 