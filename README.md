# AI Resume Analyzer

A product-style FastAPI app that compares a resume against a target job description and returns an explainable match report. It supports pasted text as well as uploaded PDF, DOCX, TXT, and Markdown resumes.

## Overview

The app is built around a simple idea: a job fit score is only useful if the person reading it can understand why it was produced.

Instead of returning a single opaque number, the analyzer breaks the comparison into:

- top matched keywords
- missing keywords
- keyword overlap ratio
- category-level scores across backend, frontend, AI/ML, engineering, and collaboration
- practical suggestions for improving alignment

## Built with

- Python
- FastAPI
- Pydantic
- PyPDF
- python-docx

## Features

- analyze pasted resume text against a pasted job description
- upload PDF, DOCX, TXT, or Markdown resumes
- extract text from common resume formats
- compute cosine-similarity-based fit scoring
- generate category breakdowns for job-relevant skills
- serve a browser UI for quick experimentation

## API

### `POST /api/analyze`
Analyze plain-text resume content.

### `POST /api/analyze-upload`
Analyze an uploaded resume file plus a pasted job description.

### `GET /api/health`
Simple health check endpoint.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open the app:

```text
http://127.0.0.1:8000/
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## Notes

This version intentionally uses transparent scoring logic instead of calling a hosted model API. That keeps the app fast to run locally and makes the output easier to inspect while still feeling like a real product workflow.
