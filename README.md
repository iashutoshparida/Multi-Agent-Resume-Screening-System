# 🤖 Agentic Resume Screening Assistant

An intelligent multi-agent system for automated resume screening with human-in-the-loop review capabilities. Built with LangGraph for orchestration and Google Gemini for LLM-powered analysis.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [What Makes This "Agentic"](#what-makes-this-agentic)
- [Agent Design](#agent-design)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Sample Inputs/Outputs](#sample-inputsoutputs)
- [Prompt Engineering](#prompt-engineering)
- [Error Handling](#error-handling)
- [Trade-offs & Assumptions](#trade-offs--assumptions)
- [Future Improvements](#future-improvements)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESUME SCREENING WORKFLOW                           │
│                            (LangGraph StateGraph)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS                                                                     │
│  ┌──────────┐    ┌─────────────────┐                                       │
│  │  Resume  │    │ Job Description │                                       │
│  │(PDF/DOCX)│    │     (Text)      │                                       │
│  └────┬─────┘    └────────┬────────┘                                       │
│       │                   │                                                 │
│       ▼                   │                                                 │
│  ┌─────────────────────┐  │                                                │
│  │  📄 RESUME PARSER   │  │         Decision Point 1:                      │
│  │      AGENT          │──┼────────► Low confidence? → Early Exit          │
│  │  (Extract structure)│  │                                                │
│  └──────────┬──────────┘  │                                                │
│             │             │                                                 │
│             ▼             ▼                                                 │
│  ┌─────────────────────────────────┐                                       │
│  │      📋 JOB ANALYZER AGENT      │  Decision Point 2:                    │
│  │    (Parse requirements)         │──► Low confidence? → Early Exit       │
│  └──────────────┬──────────────────┘                                       │
│                 │                                                           │
│       ┌─────────┴─────────┐                                                │
│       ▼                   ▼                                                 │
│  ┌──────────────┐   ┌──────────────────┐                                   │
│  │ 🎯 SKILLS    │   │ 📊 EXPERIENCE    │                                   │
│  │   MATCHER    │   │   EVALUATOR      │                                   │
│  │   AGENT      │   │     AGENT        │                                   │
│  └──────┬───────┘   └────────┬─────────┘                                   │
│         │                    │                                              │
│         └────────┬───────────┘                                              │
│                  ▼                                                          │
│  ┌─────────────────────────────────┐                                       │
│  │    🧠 DECISION MAKER AGENT      │  Decision Point 3:                    │
│  │   (Final recommendation)        │──► Low confidence? → Human Review     │
│  └──────────────┬──────────────────┘   Score variance? → Human Review      │
│                 │                                                           │
│                 ▼                                                           │
│  ┌─────────────────────────────────┐                                       │
│  │         📤 OUTPUT               │                                       │
│  │   {match_score, recommendation, │                                       │
│  │    requires_human, confidence,  │                                       │
│  │    reasoning_summary}           │                                       │
│  └─────────────────────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ What Makes This "Agentic"

### 1. **Multiple Specialized Agents**
Each agent has a clear, focused responsibility:
- Resume Parser → Extracts structured data
- Job Analyzer → Parses requirements  
- Skills Matcher → Semantic skill comparison
- Experience Evaluator → Relevance assessment
- Decision Maker → Final synthesis

### 2. **Structured Data Passing**
Agents communicate via typed Pydantic models, not raw text.

### 3. **Conditional Routing (Decision Points)**
The workflow branches based on intermediate results:
- **Decision Point 1**: If resume parsing confidence < 0.3 → Early exit
- **Decision Point 2**: If job analysis confidence < 0.3 → Early exit  
- **Decision Point 3**: If final confidence < 0.7 OR score variance > 0.3 → Human review

### 4. **Evolving State**
State accumulates as agents process - flags, confidence scores, and logs build up.

### 5. **Human-in-the-Loop Design**
System knows when to defer to humans for edge cases.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

---

## 📖 Usage

```bash
# Screen a resume against a job description
python main.py resume.pdf job_description.txt

# Save output to JSON file
python main.py resume.pdf job_description.txt -o result.json

# Verbose mode with agent logs
python main.py resume.pdf job_description.txt -v --show-logs
```

---

## 📊 Sample Outputs

### Good Fit
```json
{
  "match_score": 0.87,
  "recommendation": "Proceed to interview",
  "requires_human": false,
  "confidence": 0.91,
  "reasoning_summary": "Strong candidate exceeding requirements..."
}
```

### Poor Fit
```json
{
  "match_score": 0.12,
  "recommendation": "Reject",
  "requires_human": false,
  "confidence": 0.94,
  "reasoning_summary": "Complete skills mismatch..."
}
```

### Edge Case
```json
{
  "match_score": 0.48,
  "recommendation": "Needs manual review",
  "requires_human": true,
  "confidence": 0.62,
  "reasoning_summary": "Career changer with potential..."
}
```

---

## ⚖️ Trade-offs & Assumptions

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Sequential agent flow | Slower but debuggable | Parallel adds complexity |
| Gemini Flash model | Less capable but free | No API costs required |
| 5 separate agents | More API calls | Better modularity |
| Conservative human review | More false positives | Better to review than miss |

---

## 🔮 Future Improvements

- Batch processing for multiple resumes
- Semantic embeddings for better skill matching
- Persistent storage and API layer
- UI dashboard for HR users

---

## 📝 License

MIT License
