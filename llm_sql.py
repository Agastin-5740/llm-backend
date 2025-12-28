"""
LLM-based natural language to SQL converter using FLAN-T5 (google/flan-t5-small)

- Lazy-loads model to avoid blocking FastAPI startup (Render-safe)
- Uses LLM only for column / count intent
- Uses deterministic SQL construction
"""

from typing import List
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ===================== MODEL CONFIG =====================

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None


def get_model():
    """
    Lazy-load tokenizer and model on first request.
    Prevents Render startup timeout.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print("Loading FLAN-T5 model...")
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        print("FLAN-T5 model loaded.")

    return _tokenizer, _model


# ===================== PROMPT =====================

SYSTEM_PROMPT = """
You are an assistant that helps generate MySQL SELECT queries.

Database:
- Single table: tickets
- Columns: id, text, category, priority, status, created_at

Your job:
- Given a natural language question, suggest either:
  * a list of columns to select, or
  * a COUNT-style expression if the user is asking "how many".

Rules:
- DO NOT write full SQL.
- DO NOT include FROM, WHERE, JOIN, LIMIT.
- Output only column names or COUNT(*).
"""


def build_prompt(question: str) -> str:
    return (
        SYSTEM_PROMPT.strip()
        + "\n\nUser question: " + question.strip()
        + "\nExpression:"
    )


# =====================================================
# 🔹 KEYWORD EXTRACTION (TEXT COLUMN)
# =====================================================

def _extract_text_keywords(question: str) -> List[str]:
    stop_words = {
        "show", "me", "all", "the", "tickets", "ticket",
        "with", "that", "are", "is", "of", "for", "in",
        "high", "medium", "low",
        "priority", "status", "open", "closed",
        "technical", "billing", "general",
        "today", "yesterday", "week", "month", "this", "last"
    }

    words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z]{3,}", question)
        if w.lower() not in stop_words
    ]

    return list(set(words))


# =====================================================
# 🔹 WHERE CONDITION BUILDER
# =====================================================

def _build_conditions_from_question(question: str) -> List[str]:
    q = question.lower()
    conditions = []

    if "high" in q:
        conditions.append("priority = 'High'")
    elif "medium" in q:
        conditions.append("priority = 'Medium'")
    elif "low" in q:
        conditions.append("priority = 'Low'")

    if "technical" in q or "tech" in q:
        conditions.append("category = 'Technical'")
    if "billing" in q or "payment" in q or "refund" in q:
        conditions.append("category = 'Billing'")
    if "general" in q or "account" in q or "profile" in q:
        conditions.append("category = 'General'")

    if "open" in q:
        conditions.append("status = 'Open'")
    if "closed" in q or "resolved" in q:
        conditions.append("status = 'Closed'")

    if "today" in q:
        conditions.append("DATE(created_at) = CURDATE()")
    elif "yesterday" in q:
        conditions.append("DATE(created_at) = (CURDATE() - INTERVAL 1 DAY)")
    elif "this week" in q or "last 7 days" in q:
        conditions.append("created_at >= (CURDATE() - INTERVAL 7 DAY)")
    elif "this month" in q:
        conditions.append("created_at >= DATE_FORMAT(CURDATE(), '%Y-%m-01')")

    keywords = _extract_text_keywords(question)
    if keywords:
        likes = [f"LOWER(text) LIKE '%{kw}%'" for kw in keywords]
        conditions.append("(" + " OR ".join(likes) + ")")

    return conditions


# =====================================================
# 🔹 COLUMN EXPRESSION CLEANER
# =====================================================

def _clean_columns_expression(expr: str) -> str:
    expr = expr.strip().lower()

    if "count" in expr:
        return "COUNT(*)"

    parts = [p.strip() for p in expr.split(",")]
    if all(re.match(r"^[a-zA-Z0-9_() ]+$", p) for p in parts):
        forbidden = {"from", "where", "join", "limit", "tickets"}
        if not any(f in expr for f in forbidden):
            return ", ".join(parts)

    return "id, text, category, priority, status, created_at"


# =====================================================
# 🔹 MAIN SQL GENERATOR
# =====================================================

def generate_sql_from_question(question: str) -> str:
    q_lower = question.lower()

    tokenizer, model = get_model()

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        num_beams=4,
        early_stopping=True,
    )

    expr = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    expr = _clean_columns_expression(expr)

    is_count = "how many" in q_lower or "count" in q_lower or "count(" in expr.lower()

    if is_count:
        select_clause = "SELECT COUNT(*) AS count"
    else:
        select_clause = f"SELECT {expr}"

    conditions = _build_conditions_from_question(question)

    sql = select_clause + " FROM tickets"

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    if not is_count:
        sql += " LIMIT 50"

    return sql + ";"


# =====================================================
# 🔹 SQL EXPLANATION
# =====================================================

def explain_sql(sql: str) -> str:
    sql_lower = sql.lower()
    explanation = []

    if "count(*)" in sql_lower:
        explanation.append("This query counts the number of tickets")
    else:
        explanation.append("This query retrieves ticket details")

    if "priority = 'high'" in sql_lower:
        explanation.append("with high priority")
    if "priority = 'medium'" in sql_lower:
        explanation.append("with medium priority")
    if "priority = 'low'" in sql_lower:
        explanation.append("with low priority")

    if "category = 'technical'" in sql_lower:
        explanation.append("related to technical issues")
    if "category = 'billing'" in sql_lower:
        explanation.append("related to billing issues")
    if "category = 'general'" in sql_lower:
        explanation.append("related to general issues")

    if "status = 'open'" in sql_lower:
        explanation.append("that are currently open")
    if "status = 'closed'" in sql_lower:
        explanation.append("that are closed")

    if "limit" in sql_lower:
        explanation.append("and limits the results to 50 records")

    return " ".join(explanation).capitalize() + "."

