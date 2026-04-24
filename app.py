from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path

from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader


BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

SKILL_BUCKETS: dict[str, set[str]] = {
    "backend": {"python", "java", "fastapi", "django", "api", "sql", "mongodb", "mysql"},
    "frontend": {"javascript", "react", "html", "css", "typescript"},
    "ai_ml": {"machine", "learning", "nlp", "prompt", "llm", "data", "model"},
    "engineering": {"git", "testing", "debugging", "algorithms", "structures", "system", "design"},
    "collaboration": {"leadership", "communication", "teamwork", "mentoring", "teaching"},
}


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=50)


class ScoreBreakdown(BaseModel):
    category: str
    matched: list[str]
    missing: list[str]
    score: float


class AnalyzeResponse(BaseModel):
    fit_score: float
    keyword_overlap_ratio: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    resume_keywords: list[str]
    job_keywords: list[str]
    suggestions: list[str]
    breakdown: list[ScoreBreakdown]


AnalyzeResponse.model_rebuild()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def keyword_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    vocabulary = set(left) | set(right)
    numerator = sum(left[word] * right[word] for word in vocabulary)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def top_keywords(counter: Counter[str], limit: int = 18) -> list[str]:
    ordered = [word for word, _count in counter.most_common()]
    return ordered[:limit]


def overlap_ratio(resume_keywords: list[str], job_keywords: list[str]) -> float:
    if not job_keywords:
        return 0.0
    overlap = len(set(resume_keywords) & set(job_keywords))
    return round((overlap / len(set(job_keywords))) * 100, 2)


def category_breakdown(resume_counter: Counter[str], job_counter: Counter[str]) -> list[ScoreBreakdown]:
    breakdown: list[ScoreBreakdown] = []
    for category, keywords in SKILL_BUCKETS.items():
        required = sorted([keyword for keyword in keywords if keyword in job_counter])
        if not required:
            continue
        matched = sorted([keyword for keyword in required if keyword in resume_counter])
        missing = sorted(set(required) - set(matched))
        score = round((len(matched) / len(required)) * 100, 2) if required else 0.0
        breakdown.append(
            ScoreBreakdown(
                category=category,
                matched=matched,
                missing=missing,
                score=score,
            )
        )
    return breakdown


def build_suggestions(missing_keywords: list[str], breakdown: list[ScoreBreakdown]) -> list[str]:
    suggestions: list[str] = []
    if missing_keywords:
        suggestions.append(
            f"Add stronger evidence of {', '.join(missing_keywords[:5])} if you genuinely have that experience."
        )
    weak_categories = [item.category for item in breakdown if item.score < 50]
    if weak_categories:
        suggestions.append(
            f"Your biggest gaps for this role are in {', '.join(weak_categories[:3])}. Add projects or bullets that show proof."
        )
    suggestions.append("Mirror the language of the target role naturally in your summary, projects, and skills.")
    suggestions.append("Use bullets that show scope, tools, and outcomes instead of generic responsibility statements.")
    return suggestions


async def extract_text_from_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if suffix == ".docx":
        document = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore").strip()

    raise HTTPException(status_code=400, detail="Supported formats: PDF, DOCX, TXT, MD.")


def analyze(resume_text: str, job_description: str) -> AnalyzeResponse:
    resume_counter = keyword_counts(resume_text)
    job_counter = keyword_counts(job_description)

    similarity = cosine_similarity(resume_counter, job_counter)
    resume_keywords = top_keywords(resume_counter)
    job_keywords = top_keywords(job_counter)
    matched_keywords = sorted(set(resume_keywords) & set(job_keywords))
    missing_keywords = sorted(set(job_keywords) - set(resume_keywords))[:10]
    breakdown = category_breakdown(resume_counter, job_counter)

    return AnalyzeResponse(
        fit_score=round(similarity * 100, 2),
        keyword_overlap_ratio=overlap_ratio(resume_keywords, job_keywords),
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        resume_keywords=resume_keywords,
        job_keywords=job_keywords,
        suggestions=build_suggestions(missing_keywords, breakdown),
        breakdown=breakdown,
    )


app = FastAPI(title="AI Resume Analyzer", version="2.0.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_resume(payload: AnalyzeRequest) -> AnalyzeResponse:
    return analyze(payload.resume_text, payload.job_description)


@app.post("/api/analyze-upload", response_model=AnalyzeResponse)
async def analyze_uploaded_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
) -> AnalyzeResponse:
    resume_text = await extract_text_from_upload(resume_file)
    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Could not extract enough text from the uploaded file.")
    return analyze(resume_text, job_description)
