# 📞 AI Sales Call Evaluator for Closers

## Overview
This project analyzes sales call transcripts using AI (OpenAI GPT-4), scores closers on key criteria, extracts insights, and stores results in a cloud database for reporting and improvement tracking.

## Features
- Automatic transcript analysis and scoring
- Key insight extraction (questions, objections, sentiment)
- Structured, actionable feedback
- Historical performance tracking
- Supabase integration for data storage
- Ready for automation (n8n/Make.com)

## Setup
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run the main script or use the API endpoints

## Directory Structure
- `main.py`: Entry point
- `langchain/`: Core AI logic
- `db/`: Database models and utils
- `data/`: Sample data
- `embeddings/`: Embedding logic
- `sample_transcripts/`: Example transcripts
- `make-workflows/`: Automation scripts

## Usage
- Place transcripts in `sample_transcripts/` for testing
- Run analysis via `main.py` or API
- Results are stored in Supabase

## API Keys
See `.env.example` for required environment variables.

## Roadmap
- [x] Transcript Analyzer
- [x] AI Scoring Engine
- [x] Insight Extractor
- [x] Database + Schema
- [ ] Closer Performance Dashboard (separate project)
- [ ] Automation Workflow (n8n/Make)
- [x] Setup Guide & Documentation

---
For more details, see the project plan in the repo.

---

## 📌 Features

- �� Grades sales calls based on:
  - Rapport-building
  - Discovery
  - Objection handling
  - Pitch delivery
  - Closing effectiveness
- 🧠 Uses GPT + LangChain + Embeddings to analyze and compare with "good" calls
- 📁 Stores structured feedback in a relational DB (Supabase)
- 📈 Provides coaching-style suggestions and overall call grades
- 🔄 Automates the pipeline using Make.com

---

## 🗂️ Project Structure

```
/ai-sales-evaluator
├── data/
│   └── good_calls/        # Local folder for good call training transcripts (excluded from Git)
├── langchain/
│   └── evaluator.py       # LangChain RAG pipeline for analyzing transcripts
├── db/
│   └── schema.sql         # SQL schema for Supabase/Firebase
├── make-workflows/
│   └── webhook-handler.json
├── sample_transcripts/
│   └── sample_call.json   # Example transcript format
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Tech Stack

- **LangChain** – Embedding + Retrieval-Augmented Generation (RAG)
- **OpenAI API** – GPT-4 + Embedding Models
- **Supabase** – Cloud PostgreSQL with PGVector support
- **Make.com** – Workflow automation (trigger on Fathom transcripts)
- **Optional**: React + Tailwind CSS frontend for dashboard

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ren-418/closer-ai-feedback-langchain.git
cd closer-ai-feedback-langchain
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your_supabase_key
```

### 4. Prepare Vector Store

- Load good call transcripts into `data/good_calls`
- Run `langchain/embed_good_calls.py` to embed and store vectors

### 5. Run LangChain RAG Pipeline

```bash
python langchain/evaluator.py
```

---

## 🛑 Data Privacy Note

🔒 **Real transcripts are not included** in the repo to protect client confidentiality.

Add them locally under:

```
data/good_calls/
```

This folder is excluded via `.gitignore`.

---

## 🛠️ TODO / Roadmap

- [x] LangChain QA pipeline
- [x] Supabase schema + integration
- [ ] Fathom integration via Make.com
- [ ] Dashboard frontend (React)
- [ ] Admin controls and export options

---

## 👨‍💻 Author

**Ren** – AI Full-Stack Developer & Automation Specialist

---

## 📄 License

Private project — Not open-sourced. Usage permitted only by client or authorized team members.