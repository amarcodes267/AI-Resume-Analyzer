# 📄 AI Resume Analyzer

A lightweight, beginner-friendly **AI-powered Resume Analyzer** built with **Streamlit** and **Hugging Face Transformers**.

Upload a PDF resume and get instant analysis including resume score, skill extraction, summary, improvement suggestions, and an interactive Q&A chatbot.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📤 **Resume Upload** | Upload PDF resumes effortlessly |
| 📊 **Resume Score** | Get a 0–100 score with detailed breakdown |
| 🛠️ **Skill Extraction** | Automatic detection of technical & soft skills |
| 📋 **Resume Summary** | AI-generated summary of your resume |
| 💡 **Suggestions** | Strengths, weaknesses, and actionable recommendations |
| 💬 **Resume Chat** | Ask questions about your resume in natural language |

---

## 🧰 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Frontend + Backend (Python web framework) |
| **Hugging Face Transformers** | Lightweight AI models (distilbert) |
| **PyMuPDF (fitz)** | PDF text extraction |
| **Python 3.10+** | Core language |

---

## 📁 Project Structure

```
AI-Resume-Analyzer/
├── app.py                 # Main Streamlit dashboard
├── analyzer.py            # Resume analysis engine (score, skills, summary, suggestions, chat)
├── resume_parser.py       # PDF text extraction using PyMuPDF
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── render.yaml            # Render Blueprint deployment config
├── setup.sh               # Render build setup script
└── .streamlit/
    └── config.toml        # Streamlit server configuration
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ *First run will download Hugging Face models (~500MB). This happens once.*

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 🌐 Deploy on Render

This project is fully configured for one-click deployment on **Render**.

### Deploy Button (Easiest)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/AI-Resume-Analyzer)

> Replace `YOUR_USERNAME` with your actual GitHub username in the URL above.

### Manual Setup (via Render Dashboard)

1. **Push to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Ready for Render deployment"
   git push
   ```

2. **Go to [Render Dashboard](https://dashboard.render.com)** → Click **"New +"** → **"Web Service"**

3. **Connect your GitHub repo** and use these settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `ai-resume-analyzer` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
   | **Plan** | `Free` (or `Starter` for better performance) |

4. Click **"Create Web Service"**

Your app will be live at `https://ai-resume-analyzer.onrender.com` 🎉

> ⏱️ **First deployment takes 5–10 minutes** — Render installs dependencies and downloads Hugging Face models (~2GB). Subsequent deploys are faster due to build caching.

### Deploy via Blueprint (render.yaml)

This repo includes a `render.yaml` file for **Infrastructure as Code**. To use it:

1. Push to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **"New +"** → **"Blueprint"**
4. Connect your repo — Render will auto-detect `render.yaml` and deploy

### ⚠️ Important Notes for Render Free Tier

| Consideration | Details |
|---------------|---------|
| **RAM** | Free tier has 512MB RAM. The app may be slow to start or occasionally idle-sleep |
| **Model Download** | HuggingFace models (`torch` + `transformers`) are ~2GB. First build takes longer |
| **Cold Start** | Free tier spins down after 15 mins of inactivity. First request after idle may take ~30s |
| **Upgrade** | For better performance, upgrade to **Starter** ($7/mo) or **Professional** plans |

### Local Server

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🎯 Usage Guide

1. **Upload Resume** — Click "Browse files" and select a PDF resume.
2. **Review Summary** — Read the AI-generated resume summary.
3. **Check Skills** — See all detected technical and soft skills.
4. **View Score** — Review your resume score (0–100) with breakdown.
5. **Improve** — Read suggestions for strengths, weaknesses, and recommendations.
6. **Chat** — Ask questions like:
   - "What skills are present in my resume?"
   - "Summarize my resume."
   - "What projects are mentioned?"
   - "How can I improve my resume?"
   - "What is my resume score?"

---

## 🧠 How It Works

```
User Uploads PDF → PyMuPDF extracts text → Analyzer processes text
    ↓
├── Resume Score (rule-based: length, sections, skills, action verbs)
├── Skills Extraction (keyword matching + AI classification)
├── Resume Summary (HuggingFace summarization model)
├── Suggestions (rule-based gap analysis)
└── Resume Chat (context matching + keyword detection)
```

### Score Breakdown

| Component | Max Points | What It Measures |
|-----------|-----------|------------------|
| Length | 30 | Word count adequacy (target: 300+ words) |
| Sections | 25 | Presence of key resume sections |
| Skills | 25 | Number and variety of skills |
| Action Verbs | 20 | Use of strong action verbs |



