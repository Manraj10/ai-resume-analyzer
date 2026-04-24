# AI Resume Analyzer

A FastAPI service that compares resume text against a target job description and returns an explainable match breakdown. The goal was to build something practical and inspectable rather than a black-box score generator.

## Overview

This service takes two text inputs:

- a resume
- a job description

It then extracts keywords, computes a similarity score, measures keyword overlap, and returns suggestions that make the output easier to reason about.

## Built with

- Python
- FastAPI
- Pydantic

## What it does

- tokenizes and cleans both text inputs
- highlights matched and missing keywords
- computes a cosine-similarity-based fit score
- reports keyword overlap as a separate signal
- returns practical suggestions based on the gaps it finds

## API

### `POST /analyze`

Request body:

```json
{
  "resume_text": "Built backend APIs with FastAPI and Python...",
  "job_description": "Looking for a software engineering intern with Python, APIs, and communication skills..."
}
```

Response fields include:

- `fit_score`
- `matched_keywords`
- `missing_keywords`
- `resume_keywords`
- `job_keywords`
- `keyword_overlap_ratio`
- `suggestions`

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## Notes

This project currently uses lightweight rule-based text processing instead of embeddings or external model APIs. That makes it fast to run locally and easy to inspect, while still showing the structure of a matching workflow.
