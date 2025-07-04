from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body
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
from database.database_manager import DatabaseManager
import json
import logging

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

# Create database manager instance
db_manager = DatabaseManager()

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

class NewCallRequest(BaseModel):
    closer_name: str
    closer_email: str
    transcript_text: Optional[str] = None
    date_of_call: Optional[str] = None

class CloserEmailRequest(BaseModel):
    closer_email: str

class BusinessRuleCreate(BaseModel):
    criteria_name: str
    description: str
    violation_text: str
    correct_text: Optional[str] = None
    score_penalty: int = -2
    feedback_message: str
    category: str = "general"

class BusinessRuleUpdate(BaseModel):
    description: Optional[str] = None
    violation_text: Optional[str] = None
    correct_text: Optional[str] = None
    score_penalty: Optional[int] = None
    feedback_message: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None

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
    """Create a new closer (by email, or return existing)."""
    try:
        closer_obj = db_manager.create_closer(
            closer.name,
            closer.email,
            closer.phone,
            closer.hire_date
        )
        if not closer_obj:
            raise HTTPException(status_code=500, detail="Failed to create closer")
        return closer_obj
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

@app.post("/closers/email", dependencies=[Depends(verify_token)])
async def get_closer_by_email(request: CloserEmailRequest):
    """Get specific closer by email (POST)."""
    try:
        closer = db_manager.get_closer_by_email(request.closer_email)
        if not closer:
            raise HTTPException(status_code=404, detail="Closer not found")
        return closer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Call management endpoints
@app.get("/calls", dependencies=[Depends(verify_token)])
async def get_calls(
    closer_email: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    """Get calls with optional filtering (by closer_email, status, date)."""
    try:
        calls = db_manager.get_calls(closer_email=closer_email, status=status, start_date=start_date, end_date=end_date, limit=limit)
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

@app.post("/calls/new-call")
async def new_call(request: NewCallRequest):
    """Create and analyze a new call from JSON (for Google Sheet/automation)."""
    try:
        
        print("New call request received")
        print(request)
        # Validate required fields
        if not request.closer_name or not request.closer_email:
            raise HTTPException(status_code=400, detail="closer_name and closer_email are required.")
        if not request.transcript_text or not request.transcript_text.strip():
            raise HTTPException(status_code=400, detail="transcript_text is required and cannot be empty.")
        # Optionally validate date_of_call format if needed
        db_manager.create_closer(request.closer_name, request.closer_email)
        call_record = db_manager.create_call(
            closer_name=request.closer_name,
            closer_email=request.closer_email,
            transcript_text=request.transcript_text,
            call_date=request.date_of_call
        )
        if not call_record:
            raise HTTPException(status_code=500, detail="Failed to create call record")
        analysis_result = evaluator.evaluate_transcript(request.transcript_text)
        if analysis_result.get('status') == 'failed':
            raise HTTPException(status_code=500, detail=analysis_result.get('error', 'Analysis failed'))
        success = db_manager.update_call_analysis(call_record['id'], analysis_result)   
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store analysis results")
        return {
            "status": "success",
            "call_id": call_record['id']
        }
    except HTTPException as e:
        raise e
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

# Business Rules Management endpoints
# @app.get("/business-rules", dependencies=[Depends(verify_token)])
# async def get_business_rules():
#     """Get all active business rules."""
#     try:
#         rules = db_manager.get_business_rules()
#         return {"business_rules": rules}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/business-rules", dependencies=[Depends(verify_token)])
# async def create_business_rule(rule: BusinessRuleCreate):
#     """Create a new business rule."""
#     try:
#         rule_obj = db_manager.create_business_rule(
#             criteria_name=rule.criteria_name,
#             description=rule.description,
#             violation_text=rule.violation_text,
#             correct_text=rule.correct_text,
#             score_penalty=rule.score_penalty,
#             feedback_message=rule.feedback_message,
#             category=rule.category
#         )
#         if not rule_obj:
#             raise HTTPException(status_code=500, detail="Failed to create business rule")
#         return rule_obj
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.put("/business-rules/{rule_id}", dependencies=[Depends(verify_token)])
# async def update_business_rule(rule_id: str, rule: BusinessRuleUpdate):
#     """Update an existing business rule."""
#     try:
#         rule_obj = db_manager.update_business_rule(rule_id, rule.dict(exclude_unset=True))
#         if not rule_obj:
#             raise HTTPException(status_code=404, detail="Business rule not found")
#         return rule_obj
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.delete("/business-rules/{rule_id}", dependencies=[Depends(verify_token)])
# async def delete_business_rule(rule_id: str):
#     """Delete a business rule."""
#     try:
#         success = db_manager.delete_business_rule(rule_id)
#         if not success:
#             raise HTTPException(status_code=404, detail="Business rule not found")
#         return {"status": "success", "message": "Business rule deleted"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        # Test DB connection on startup
        test_db = db_manager.get_all_closers()
        print(f"✅ Connected to Supabase DB. Found {len(test_db)} closers.")
    except Exception as e:
        print(f"❌ Could not connect to Supabase DB: {e}")
    print("🚀 API server is starting on http://0.0.0.0:5000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True) 