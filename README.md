# 📞 AI Sales Call Evaluator for Closers

An AI-powered system that evaluates sales call transcripts from closers. The system analyzes transcripts using GPT, identifies strengths and weaknesses, and provides structured feedback to improve closing performance.

---

## 📌 Features

- 📊 Grades sales calls based on:
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