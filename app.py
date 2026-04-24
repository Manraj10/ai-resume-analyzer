from __future__ import annotations

import math
import re
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel, Field


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

PRIORITY_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "react",
    "fastapi",
    "django",
    "sql",
    "aws",
    "git",
    "api",
    "machine",
    "learning",
    "data",
    "structures",
    "algorithms",
    "leadership",
    "communication",
}


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=50)


class AnalyzeResponse(BaseModel):
    fit_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    resume_keywords: list[str]
    job_keywords: list[str]
    suggestions: list[str]


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


def top_keywords(counter: Counter[str], limit: int = 15) -> list[str]:
    priority = [word for word in counter if word in PRIORITY_KEYWORDS]
    regular = [word for word, _count in counter.most_common() if word not in PRIORITY_KEYWORDS]
    ordered = sorted(priority, key=lambda word: (-counter[word], word)) + regular
    return ordered[:limit]


def build_suggestions(missing_keywords: list[str]) -> list[str]:
    suggestions: list[str] = []
    if missing_keywords:
        suggestions.append(
            f"Add stronger evidence of {', '.join(missing_keywords[:5])} if you genuinely have that experience."
        )
    suggestions.append("Use project bullets that mention measurable outcomes, technical scope, and tools.")
    suggestions.append("Mirror the language of the target role naturally in your summary and skills section.")
    return suggestions


app = FastAPI(title="AI Resume Analyzer", version="1.0.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Resume Analyzer is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(payload: AnalyzeRequest) -> AnalyzeResponse:
    resume_counter = keyword_counts(payload.resume_text)
    job_counter = keyword_counts(payload.job_description)

    similarity = cosine_similarity(resume_counter, job_counter)
    resume_keywords = top_keywords(resume_counter)
    job_keywords = top_keywords(job_counter)

    matched_keywords = sorted(set(resume_keywords) & set(job_keywords))
    missing_keywords = sorted(set(job_keywords) - set(resume_keywords))[:10]

    fit_score = round(similarity * 100, 2)

    return AnalyzeResponse(
        fit_score=fit_score,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        resume_keywords=resume_keywords,
        job_keywords=job_keywords,
        suggestions=build_suggestions(missing_keywords),
    )
