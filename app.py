"""
app.py

Main Streamlit dashboard for the AI Resume Analyzer.

Sections:
  1. Upload Resume (PDF)
  2. Resume Summary
  3. Skills Extracted
  4. Resume Score (with metrics + progress bar)
  5. Suggestions (Strengths, Weaknesses, Recommendations)
  6. Ask Questions (Resume Chat)
"""

import streamlit as st
import io

# ── Local imports ──
from resume_parser import extract_text_from_pdf, count_pages
from analyzer import (
    calculate_resume_score,
    extract_skills,
    generate_summary,
    generate_suggestions,
    chat_with_resume,
)

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
        /* ── Main Container ── */
        .main { padding: 1rem 2rem; }

        /* ── Header ── */
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .header-subtitle {
            font-size: 1rem;
            color: #a0a0a0;
            margin-bottom: 2rem;
        }

        /* ── Cards ── */
        .card {
            background: #1e1e2e;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #2d2d44;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #c0c0ff;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-content {
            font-size: 0.95rem;
            color: #d0d0d0;
            line-height: 1.6;
        }

        /* ── Score Card ── */
        .score-number {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
        }
        .score-grade {
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
        }

        /* ── Skill Tags ── */
        .skill-tag {
            display: inline-block;
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border: 1px solid #667eea44;
            color: #c0c0ff;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.2rem;
        }

        /* ── Suggestion Cards ── */
        .strength-text { color: #4ade80; }
        .weakness-text { color: #fb923c; }
        .suggestion-text { color: #60a5fa; }

        /* ── Chat ── */
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        .user-message {
            background: #2d2d44;
            border-left: 3px solid #667eea;
        }
        .bot-message {
            background: #1e1e2e;
            border-left: 3px solid #4ade80;
        }

        /* ── Divider ── */
        .divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #667eea44, transparent);
            margin: 2rem 0;
        }

        /* Streamlit overrides */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
        }
        .stButton > button:hover {
            opacity: 0.9;
            color: white;
        }
        .stTextInput > div > div > input {
            border-radius: 8px;
            background: #1e1e2e;
            border: 1px solid #2d2d44;
            color: white;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# HELPER (defined early so it's available when called below)
# ══════════════════════════════════════════════════════════════════════

def _get_max_score(key: str) -> int:
    """Return the maximum possible score for a breakdown category."""
    max_scores = {
        "length": 30,
        "sections": 25,
        "skills": 25,
        "action_verbs": 20,
    }
    return max_scores.get(key, 25)


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div class="header-title">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtitle">Upload your resume (PDF) and get instant AI-powered analysis, '
    "score, and improvement suggestions.</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: UPLOAD RESUME
# ══════════════════════════════════════════════════════════════════════

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Only PDF files are supported.",
    )

with col2:
    if uploaded_file is not None:
        st.info(f"📎 **{uploaded_file.name}**", icon=None)

# ── Process uploaded file ──
if uploaded_file is not None and not st.session_state.analysis_done:
    with st.spinner("🔍 Analyzing your resume..."):
        # Read the file bytes
        pdf_bytes = io.BytesIO(uploaded_file.read())

        # Extract text
        resume_text = extract_text_from_pdf(pdf_bytes)

        # Count pages
        pdf_bytes.seek(0)
        page_count = count_pages(pdf_bytes)

        # Store in session state
        st.session_state.resume_text = resume_text
        st.session_state.page_count = page_count
        st.session_state.analysis_done = True
        st.session_state.file_name = uploaded_file.name

        st.rerun()

elif uploaded_file is not None and st.session_state.analysis_done:
    # Allow re-upload — reset if file changes
    if st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.analysis_done = False
        st.session_state.resume_text = ""
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# DISPLAY ANALYSIS RESULTS
# ══════════════════════════════════════════════════════════════════════

if st.session_state.analysis_done and st.session_state.resume_text:
    resume_text = st.session_state.resume_text

    # ── Run analysis (cached) ──
    @st.cache_data(show_spinner=False)
    def run_analysis(text: str):
        score_data = calculate_resume_score(text)
        skills = extract_skills(text)
        summary = generate_summary(text)
        suggestions = generate_suggestions(text, skills)
        return score_data, skills, summary, suggestions

    score_data, skills, summary, suggestions = run_analysis(resume_text)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 2: RESUME SUMMARY
    # ════════════════════════════════════════════════════════════════════

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown("## 📋 Resume Summary")
    st.markdown(
        f'<div class="card"><div class="card-content">{summary}</div></div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3: SKILLS EXTRACTED
    # ════════════════════════════════════════════════════════════════════

    st.markdown("## 🛠️ Skills Extracted")
    if skills:
        skills_html = "".join(
            f'<span class="skill-tag">{skill}</span>' for skill in skills
        )
        st.markdown(
            f'<div class="card"><div class="card-content">{skills_html}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="card"><div class="card-content">⚠️ No specific skills detected. '
            "Consider adding a Skills section with relevant keywords.</div></div>",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # SECTION 4: RESUME SCORE
    # ════════════════════════════════════════════════════════════════════

    st.markdown("## 📊 Resume Score")

    score = score_data["score"]
    grade = score_data["grade"]
    breakdown = score_data["breakdown"]
    word_count = score_data.get("word_count", 0)

    col_score1, col_score2, col_score3, col_score4 = st.columns(4)

    with col_score1:
        st.metric(label="📝 Word Count", value=f"{word_count}")

    with col_score2:
        st.metric(label="🏆 Score", value=f"{score}/100")

    with col_score3:
        st.metric(label="🎓 Grade", value=grade)

    with col_score4:
        skills_found = len(skills) if skills else 0
        st.metric(label="🔧 Skills Found", value=f"{skills_found}")

    # Progress bar
    st.progress(score / 100)

    # Breakdown
    st.markdown("**Score Breakdown:**")
    breakdown_cols = st.columns(len(breakdown))
    for idx, (key, value) in enumerate(breakdown.items()):
        label = key.replace("_", " ").title()
        with breakdown_cols[idx]:
            st.caption(label)
            st.markdown(f"**{value} / {_get_max_score(key)}**")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 5: SUGGESTIONS
    # ════════════════════════════════════════════════════════════════════

    st.markdown("## 💡 Suggestions")

    sug_col1, sug_col2, sug_col3 = st.columns(3)

    with sug_col1:
        st.markdown("### ✅ Strengths")
        if suggestions["strengths"]:
            for s in suggestions["strengths"]:
                st.markdown(f'<div class="card"><span class="strength-text">✅ {s}</span></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card"><span class="suggestion-text">No specific strengths identified yet.</span></div>',
                unsafe_allow_html=True,
            )

    with sug_col2:
        st.markdown("### ⚠️ Areas to Improve")
        if suggestions["weaknesses"]:
            for w in suggestions["weaknesses"]:
                st.markdown(f'<div class="card"><span class="weakness-text">⚠️ {w}</span></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card"><span class="strength-text">✨ No major issues found!</span></div>',
                unsafe_allow_html=True,
            )

    with sug_col3:
        st.markdown("### 🚀 Recommendations")
        if suggestions["suggestions"]:
            for s in suggestions["suggestions"]:
                st.markdown(f'<div class="card"><span class="suggestion-text">💡 {s}</span></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card"><span class="strength-text">✨ Your resume looks great!</span></div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════
    # SECTION 6: ASK QUESTIONS (RESUME CHAT)
    # ════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("## 💬 Ask Questions About Your Resume")

    st.markdown(
        '<div class="card" style="background: #16162a;">'
        '<div class="card-title">🤖 Resume Assistant</div>'
        '<div class="card-content">'
        "Ask me anything about your resume! Examples: "
        '<em>"What skills do I have?"</em>, '
        '<em>"Summarize my resume"</em>, '
        '<em>"How can I improve?"</em>, '
        '<em>"What projects are mentioned?"</em>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Chat input
    user_query = st.text_input(
        "Your question:",
        placeholder="e.g., What skills are in my resume?",
        key="chat_input",
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_query:
        with st.spinner("Thinking..."):
            # Get response
            bot_response = chat_with_resume(user_query, resume_text)

            # Add to history
            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("bot", bot_response))

            # Clear input — but streamlit reruns, so we use a flag
            st.session_state.last_query = user_query

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>🧑 You:</strong> {message}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-message bot-message">'
                    f'<strong>🤖 Assistant:</strong> {message}</div>',
                    unsafe_allow_html=True,
                )

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Reset / Analyze New Resume ──
    st.markdown("---")
    if st.button("🔄 Analyze New Resume", type="secondary"):
        for key in ["resume_text", "analysis_done", "file_name", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

else:
    # ── Initial state: no resume uploaded ──
    st.markdown("---")
    st.markdown(
        '<div class="card" style="text-align: center; padding: 3rem;">'
        '<div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>'
        '<div class="card-title" style="font-size: 1.3rem;">Upload Your Resume to Get Started</div>'
        '<div class="card-content">'
        "Upload a PDF resume above and I'll analyze it for you.<br>"
        "You'll get a score, skill extraction, summary, and suggestions!"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">'
    "AI Resume Analyzer · Built with Streamlit &amp; Hugging Face Transformers"
    "</div>",
    unsafe_allow_html=True,
)




