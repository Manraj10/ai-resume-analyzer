# AI Resume Analyzer

A recruiter-facing backend application that scores how well a resume matches a job description using lightweight NLP and transparent scoring logic. This project is designed to demonstrate backend engineering, text processing, API design, and explainable ranking rather than a black-box demo.

## Why this project is strong

This project shows:

- API-first backend development with FastAPI
- text preprocessing and keyword extraction
- similarity scoring between resumes and job descriptions
- explainable outputs recruiters or hiring teams can actually inspect

## Tech stack

- Python
- FastAPI
- Pydantic

## Features

- upload or paste resume text
- paste a target job description
- extract important technical and soft-skill keywords
- compute an overall fit score
- return matched keywords, missing keywords, and improvement suggestions

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Resume-ready description

Built a FastAPI-based resume analysis tool that compares resumes against job descriptions using keyword extraction and similarity scoring, returning explainable match insights such as matched skills, missing keywords, and improvement suggestions.
