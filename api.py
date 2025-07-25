from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body, BackgroundTasks
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
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Sales Call Evaluator API",
    description="Professional API for evaluating sales calls with business management features",
    version="2.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-sales-call-evaluator.vercel.app","http://localhost:5173" ,"https://vote-mentioned-keeping-indonesia.trycloudflare.com"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 365 * 100 
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
    lead_name: str
    transcript_link:str
    transcript_text: Optional[str] = None
    date_of_call: Optional[str] = None

class CloserEmailRequest(BaseModel):
    closer_email: str

class MarkAsReadRequest(BaseModel):
    call_ids: List[str]

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
    except jwt.PyJWTError:
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

@app.delete("/closers/remove", dependencies=[Depends(verify_token)])
async def remove_closer(request: CloserEmailRequest):
    """Remove a closer by email (soft delete)."""
    try:
        # First check if closer exists
        closer = db_manager.get_closer_by_email(request.closer_email)
        if not closer:
            raise HTTPException(status_code=404, detail="Closer not found")
        
        # Remove the closer
        success = db_manager.remove_closer(request.closer_email)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove closer")
        
        return {
            "status": "success",
            "message": f"Closer {request.closer_email} has been removed",
            "removed_closer": {
                "email": request.closer_email,
                "name": closer.get('name', 'Unknown')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Call management endpoints
@app.get("/calls", dependencies=[Depends(verify_token)])
async def get_calls(
    closer_email: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    current_user: Dict = Depends(verify_token)
):
    """Get calls with optional filtering (by closer_email, status, date) and include read status for current admin."""
    try:
        admin_email = current_user.get('email')
        calls = db_manager.get_calls(
            closer_email=closer_email, 
            status=status, 
            start_date=start_date, 
            end_date=end_date, 
            limit=limit,
            admin_email=admin_email
        )
        return {"calls": calls, "total": len(calls)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calls/unread-count", dependencies=[Depends(verify_token)])
async def get_unread_calls_count(current_user: Dict = Depends(verify_token)):
    """Get count of unread analyzed calls for the current admin."""
    try:
        admin_email = current_user.get('email')
        if not admin_email:
            raise HTTPException(status_code=400, detail="Admin email not found in token")
        
        count = db_manager.get_unread_calls_count(admin_email)
        return {
            "unread_count": count,
            "admin_email": admin_email
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calls/{call_id}", dependencies=[Depends(verify_token)])
async def get_call(call_id: str, current_user: Dict = Depends(verify_token)):
    """Get specific call with full analysis and read status for current admin."""
    try:
        admin_email = current_user.get('email')
        call = db_manager.get_call_by_id(call_id, admin_email)
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        return call
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calls/mark-as-read", dependencies=[Depends(verify_token)])
async def mark_calls_as_read(request: MarkAsReadRequest, current_user: Dict = Depends(verify_token)):
    """Mark specific calls as read for the current admin."""
    try:
        if not request.call_ids:
            raise HTTPException(status_code=400, detail="call_ids list cannot be empty")
        
        # Use admin email from token for security
        admin_email = current_user.get('email')
        if not admin_email:
            raise HTTPException(status_code=400, detail="Admin email not found in token")
        
        success = db_manager.mark_calls_as_read(admin_email, request.call_ids)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to mark calls as read")
        
        return {
            "status": "success",
            "message": f"Marked {len(request.call_ids)} call(s) as read",
            "marked_calls": request.call_ids,
            "admin_email": admin_email
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/calls/{call_id}", dependencies=[Depends(verify_token)])
async def delete_call(call_id: str):
    """Delete a specific call and all its related data."""
    try:
        # First check if call exists
        call = db_manager.get_call_by_id(call_id)
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Delete the call (this should cascade to related tables due to foreign key constraints)
        success = db_manager.delete_call(call_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete call")
        
        return {
            "status": "success",
            "message": f"Call {call_id} has been deleted",
            "deleted_call": {
                "id": call_id,
                "closer_name": call.get('closer_name', 'Unknown'),
                "closer_email": call.get('closer_email', 'Unknown'),
                "lead_name": call.get('lead_name', 'Unknown')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/leaderboard", dependencies=[Depends(verify_token)])
async def get_leaderboard():
    """Get closers leaderboard."""
    try:
        analytics = db_manager.get_team_analytics()
        return {
            "leaderboard": analytics.get('leaderboard', []),
            "top_performers": analytics.get('top_performers', []),
            "coaching_insights": analytics.get('coaching_insights', {}),
            "team_average": analytics.get('team_average', 0),
            "total_calls": analytics.get('total_calls', 0),
            "total_analyzed_calls": analytics.get('total_analyzed_calls', 0),
            "total_average_score": analytics.get('total_average_score', 0),
            "period_metrics": analytics.get('period_metrics', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calls/new-call")
async def new_call(request: NewCallRequest, background_tasks: BackgroundTasks):
    """Create and analyze a new call from JSON (for Google Sheet/automation). Accepts transcript_text as base64-encoded string. Responds immediately and processes analysis in the background."""
    try:
        logging.info("New call request received: %s", request)
        # Validate required fields
        if not request.closer_name or not request.closer_email:
            logging.warning("Missing closer_name or closer_email in request: %s", request)
            return {"status": "error", "detail": "closer_name and closer_email are required."}
        if not request.transcript_text or not request.transcript_text.strip():
            logging.warning("Missing or empty transcript_text in request: %s", request)
            return {"status": "error", "detail": "transcript_text is required and cannot be empty."}
        # Decode base64 transcript
        try:
            decoded_bytes = base64.b64decode(request.transcript_text)
            transcript_decoded = decoded_bytes.decode('utf-8')
        except Exception as decode_err:
            logging.error("Failed to decode transcript_text from base64: %s", decode_err)
            return {"status": "error", "detail": "transcript_text must be valid base64-encoded UTF-8 text."}
        # Optionally validate date_of_call format if needed
        db_manager.create_closer(request.closer_name, request.closer_email)
        call_record = db_manager.create_call(
            closer_name=request.closer_name,
            closer_email=request.closer_email,
            lead_name = request.lead_name,
            transcript_link = request.transcript_link,
            transcript_text=transcript_decoded,
            call_date=request.date_of_call
        )
        if not call_record:
            logging.error("Failed to create call record for: %s", request)
            return {"status": "error", "detail": "Failed to create call record"}
        # Launch background analysis and notification
        background_tasks.add_task(analyze_and_notify_make, call_record['id'], request.closer_name, request.closer_email,request.lead_name, transcript_decoded,request.transcript_link)
        logging.info("Call created, analysis scheduled in background: %s", call_record['id'])
        return {
            "status": "pending",
            "call_id": call_record['id']
        }
    except Exception as e:
        logging.exception("Unexpected error in new_call endpoint")
        return {"status": "error", "detail": "Internal server error."}

# Helper function for background analysis and Make.com notification
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "https://hook.us2.make.com/pk6rsf6dyy6wdx8lwenirqiewjpggqj2")

def analyze_and_notify_make(call_id, closer_name, closer_email, lead_name,transcript_decoded,transcript_link):
    try:
        analysis_result = evaluator.evaluate_transcript(transcript_decoded)
        db_manager.update_call_analysis(call_id, analysis_result)
        payload = {
            "call_id": call_id,
            "closer_name": closer_name,
            "closer_email": closer_email,
            "lead_name": lead_name,
            "transcript_link": transcript_link,
            "status": "success"
        }
        try:
            requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=10)
            logging.info(f"Successfully notified Make.com webhook for call_id: {call_id}")
        except Exception as notify_err:
            logging.error(f"Failed to notify Make.com webhook: {notify_err}")
    except Exception as e:
        logging.error(f"Background analysis or notification failed for call_id {call_id}: {e}")

# Enable business rules management endpoints
@app.get("/business-rules", dependencies=[Depends(verify_token)])
async def get_business_rules():
    """Get all active business rules."""
    try:
        rules = db_manager.get_business_rules()
        return {"business_rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/business-rules", dependencies=[Depends(verify_token)])
async def create_business_rule(rule: BusinessRuleCreate):
    """Create a new business rule."""
    try:
        rule_obj = db_manager.create_business_rule(
            criteria_name=rule.criteria_name,
            description=rule.description,
            violation_text=rule.violation_text,
            correct_text=rule.correct_text,
            score_penalty=rule.score_penalty,
            feedback_message=rule.feedback_message,
            category=rule.category
        )
        if not rule_obj:
            raise HTTPException(status_code=500, detail="Failed to create business rule")
        return rule_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/business-rules/{rule_id}", dependencies=[Depends(verify_token)])
async def update_business_rule(rule_id: str, rule: BusinessRuleUpdate):
    """Update an existing business rule."""
    try:
        rule_obj = db_manager.update_business_rule(rule_id, rule.dict(exclude_unset=True))
        if not rule_obj:
            raise HTTPException(status_code=404, detail="Business rule not found")
        return rule_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/business-rules/{rule_id}", dependencies=[Depends(verify_token)])
async def delete_business_rule(rule_id: str):
    """Delete a business rule."""
    try:
        success = db_manager.delete_business_rule(rule_id)
        if not success:
            raise HTTPException(status_code=404, detail="Business rule not found")
        return {"status": "success", "message": "Business rule deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        # Test DB connection on startup
        test_db = db_manager.get_all_closers()
        print(f"✅ Connected to Supabase DB. Found {len(test_db)} closers.")
    except Exception as e:
        print(f"❌ Could not connect to Supabase DB: {e}")
    print("🚀 API server is starting on http://0.0.0.0:5000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True) 