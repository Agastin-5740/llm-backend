from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db, engine
from llm_sql import generate_sql_from_question
from llm_sql import explain_sql   # 👈 NEW

app = FastAPI(title="AI Ticket Analytics")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class NLQueryRequest(BaseModel):
    question: str

class Insights(BaseModel):
    total_records: int

class NLQueryResponse(BaseModel):
    sql: str
    explanation: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    insights: Insights

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"message": "AI Ticket Analytics API running"}

@app.post("/nl-query", response_model=NLQueryResponse)
def nl_query(req: NLQueryRequest, db: Session = Depends(get_db)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1️⃣ Generate SQL
    try:
        sql = generate_sql_from_question(req.question)
        print("Generated SQL:", sql)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM SQL error: {str(e)}")

    if not sql.lower().strip().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

    # 2️⃣ Explain SQL (NEW)
    try:
        explanation = explain_sql(sql)
    except Exception:
        explanation = "This query retrieves ticket data based on the given conditions."

    # 3️⃣ Execute SQL
    try:
        result = db.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {str(e)}")

    insights = {
        "total_records": len(rows)
    }

    return {
        "sql": sql,
        "explanation": explanation,
        "columns": columns,
        "rows": rows,
        "insights": insights
    }
