"""
analyzer.py

Core analysis engine for the AI Resume Analyzer.

Provides:
  - Resume Scoring (rule-based)
  - Skill Extraction (keyword-based + AI-enhanced)
  - Resume Summarization (extractive + AI)
  - Improvement Suggestions (rule-based gap analysis)
  - Q&A Chat (context matching against resume text)

Uses Hugging Face Transformers with a lightweight model (distilbert).
"""

import re
import string
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────
# SKILLS DATABASE (Lightweight keyword-based approach)
# ──────────────────────────────────────────────────────────────────────

SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust",
    "swift", "kotlin", "php", "scala", "r", "matlab", "perl", "dart",
    # Web Technologies
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring", "bootstrap", "tailwind", "jquery", "sass", "graphql",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis", "elasticsearch",
    "firebase", "cassandra", "dynamodb", "mariadb",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins", "git",
    "github actions", "ci/cd", "terraform", "ansible", "prometheus", "grafana",
    # Data Science & AI
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "natural language processing", "computer vision", "data science", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv",
    "llm", "large language model", "rag", "generative ai", "langchain",
    # Data & Analytics
    "tableau", "power bi", "excel", "looker", "spark", "hadoop", "airflow", "snowflake",
    # Mobile Development
    "android", "ios", "flutter", "react native", "xamarin",
    # Soft Skills
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "project management", "agile", "scrum", "time management",
    # Other Technical
    "linux", "unix", "bash", "powershell", "rest api", "api", "microservices",
    "testing", "unit test", "selenium", "jira", "confluence",
    # Certifications / Domains
    "pmp", "scrum master", "aws certified", "azure certified",
    "blockchain", "cybersecurity", "iot", "robotics", "arduino", "raspberry pi",
]

# Section keywords for scoring
SECTION_KEYWORDS = {
    "experience": ["experience", "work history", "employment", "professional experience",
                   "work experience"],
    "education": ["education", "academic", "degree", "university", "college",
                  "bachelor", "master", "phd", "school"],
    "skills": ["skills", "technical skills", "core competencies", "expertise",
               "technologies"],
    "projects": ["projects", "project experience", "personal projects"],
    "summary": ["summary", "profile", "objective", "about me", "professional summary"],
}


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize text: lowercase, remove extra whitespace."""
    text = text.lower().strip()
    # Remove bullet points and special chars but keep alphanumeric and spaces
    text = re.sub(r'[•●▪■➢–—]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())


def _detect_sections(text: str) -> Dict[str, bool]:
    """Detect which resume sections are present based on keyword matching."""
    cleaned = _clean_text(text)
    detected = {}
    for section, keywords in SECTION_KEYWORDS.items():
        detected[section] = any(kw in cleaned for kw in keywords)
    return detected


def _extract_section_text(text: str, section_name: str) -> str:
    """Roughly extract text under a detected section heading."""
    cleaned = _clean_text(text)
    keywords = SECTION_KEYWORDS.get(section_name, [section_name])

    lines = cleaned.split('\n')
    capturing = False
    section_lines = []
    section_keywords_flat = set(SECTION_KEYWORDS.keys())

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if this line is a section header
        is_section_header = False
        for kw in keywords:
            # Check if line starts with or contains the keyword as a section
            if kw in line_stripped and len(line_stripped) < 60:
                capturing = True
                is_section_header = True
                break

        if is_section_header:
            continue

        # Check if we hit another section
        if capturing:
            # Stop at next section
            next_section = False
            for sk in section_keywords_flat:
                if sk == section_name:
                    continue
                for kw in SECTION_KEYWORDS[sk]:
                    if kw in line_stripped and len(line_stripped) < 60:
                        next_section = True
                        break
                if next_section:
                    break
            if next_section:
                break
            section_lines.append(line_stripped)

    return ' '.join(section_lines)


# ──────────────────────────────────────────────────────────────────────
# AI-ENHANCED ANALYSIS (HuggingFace)
# ──────────────────────────────────────────────────────────────────────

# Lazy-loaded pipelines
_summarizer = None
_classifier = None
_text_gen = None


def _get_summarizer():
    """Lazy load the summarization pipeline."""
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            _summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-6-6",
                device=-1  # Use CPU
            )
        except Exception:
            _summarizer = False
    return _summarizer if _summarizer is not False else None


def _get_classifier():
    """Lazy load the zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
        except Exception:
            _classifier = False
    return _classifier if _classifier is not False else None


def _get_text_gen():
    """
    Lazy load a lightweight instruction-tuned text generation model
    for answering resume questions. Uses google/flan-t5-small (~80MB).
    """
    global _text_gen
    if _text_gen is None:
        try:
            from transformers import pipeline
            _text_gen = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,
                max_length=200,
                do_sample=False,
            )
        except Exception:
            _text_gen = False
    return _text_gen if _text_gen is not False else None


# ──────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def calculate_resume_score(text: str) -> Dict:
    """
    Calculate a resume score based on multiple criteria.

    Returns a dict with:
      - score (int): 0-100
      - breakdown (dict): individual component scores
      - grade (str): letter grade
    """
    if not text or text.startswith("[Error"):
        return {"score": 0, "breakdown": {}, "grade": "F"}

    word_count = _count_words(text)
    sections = _detect_sections(text)
    skills = extract_skills(text)
    cleaned = _clean_text(text)

    breakdown = {}

    # 1. Length Score (30 points) — ideal: 300+ words
    length_score = min(30, (word_count / 300) * 30)
    breakdown["length"] = round(length_score, 1)

    # 2. Section Coverage (25 points) — 5 sections, 5 pts each
    section_score = sum(5 for s, found in sections.items() if found)
    breakdown["sections"] = section_score

    # 3. Skills Score (25 points) — based on number of skills found
    skill_count = len(skills)
    skills_score = min(25, skill_count * 2.5)
    breakdown["skills"] = round(skills_score, 1)

    # 4. Keyword Density / Quality (20 points)
    # Check for action verbs, metrics, quantifiable results
    action_verbs = [
        "developed", "managed", "led", "created", "designed", "implemented",
        "improved", "increased", "decreased", "reduced", "achieved", "delivered",
        "built", "optimized", "architected", "coordinated", "established",
        "generated", "launched", "negotiated", "performed", "planned",
        "produced", "recommended", "solved", "trained", "transformed",
    ]
    action_count = sum(1 for verb in action_verbs if verb in cleaned)
    action_score = min(20, action_count * 2)
    breakdown["action_verbs"] = round(action_score, 1)

    total_score = length_score + section_score + skills_score + action_score
    total_score = max(0, min(100, total_score))

    # Grade
    if total_score >= 90:
        grade = "A"
    elif total_score >= 80:
        grade = "B"
    elif total_score >= 70:
        grade = "C"
    elif total_score >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": round(total_score, 1),
        "breakdown": breakdown,
        "grade": grade,
        "word_count": word_count,
    }


def extract_skills(text: str) -> List[str]:
    """
    Extract skills from resume text using keyword matching.

    Returns a sorted list of unique skills found.
    """
    if not text or text.startswith("[Error"):
        return []

    cleaned = _clean_text(text)
    found_skills = set()

    for skill in SKILLS_DB:
        # Use word boundary matching for single-word skills
        if " " in skill:
            # Multi-word skill
            if skill in cleaned:
                found_skills.add(skill)
        else:
            # Single word — check with word boundaries
            if re.search(r'\b' + re.escape(skill) + r'\b', cleaned):
                found_skills.add(skill)

    return sorted(found_skills, key=lambda s: s.lower())


def generate_summary(text: str, max_length: int = 130) -> str:
    """
    Generate a summary of the resume.

    First attempts HuggingFace summarization. Falls back to extractive
    (first meaningful sentences) if AI is unavailable.
    """
    if not text or text.startswith("[Error"):
        return "No resume text available to summarize."

    summarizer = _get_summarizer()

    # Limit input to first 1024 tokens for speed
    input_text = text[:2000]

    if summarizer:
        try:
            result = summarizer(
                input_text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            if result and len(result) > 0:
                return result[0]['summary_text'].strip()
        except Exception:
            pass  # Fall through to extractive method

    # Fallback: extractive summarization (first few meaningful sentences)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    meaningful = [s.strip() for s in sentences if len(s.strip().split()) > 4]

    if not meaningful:
        # Return first 200 chars
        return text[:200].strip() + ("..." if len(text) > 200 else "")

    summary = " ".join(meaningful[:3])
    if len(summary) > 400:
        summary = summary[:397] + "..."
    return summary


def generate_suggestions(text: str, skills: List[str]) -> Dict[str, List[str]]:
    """
    Generate improvement suggestions based on resume analysis.

    Returns a dict with:
      - strengths: list of positive observations
      - weaknesses: list of areas for improvement
      - suggestions: list of actionable recommendations
    """
    if not text or text.startswith("[Error"):
        return {
            "strengths": ["Resume text could not be parsed."],
            "weaknesses": ["Unable to analyze."],
            "suggestions": ["Upload a valid PDF resume."],
        }

    word_count = _count_words(text)
    sections = _detect_sections(text)
    cleaned = _clean_text(text)

    strengths = []
    weaknesses = []
    suggestions = []

    # ── Length Analysis ──
    if word_count >= 300:
        strengths.append(f"Good resume length ({word_count} words) — detailed enough.")
    elif word_count >= 150:
        strengths.append(f"Adequate length ({word_count} words).")
        suggestions.append(f"Consider expanding your resume to 300+ words for more detail.")
    else:
        weaknesses.append(f"Resume is too short ({word_count} words).")
        suggestions.append("Add more details about your experience, skills, and projects.")

    # ── Section Coverage ──
    missing_sections = [s for s, found in sections.items() if not found]
    present_sections = [s for s, found in sections.items() if found]

    if len(present_sections) >= 4:
        strengths.append(f"Strong section coverage: {', '.join(present_sections)}.")
    elif len(present_sections) >= 2:
        strengths.append(f"Has key sections: {', '.join(present_sections)}.")
        for ms in missing_sections:
            suggestions.append(f"Consider adding a '{ms.title()}' section to your resume.")
    else:
        weaknesses.append("Very few standard resume sections detected.")
        suggestions.append("Add dedicated sections for Experience, Education, Skills, and Projects.")

    # ── Skills Analysis ──
    if len(skills) >= 15:
        strengths.append(f"Strong skill set with {len(skills)} skills listed.")
    elif len(skills) >= 8:
        strengths.append(f"Good skill coverage with {len(skills)} skills.")
        suggestions.append("Consider adding more relevant skills to strengthen your profile.")
    elif len(skills) >= 3:
        weaknesses.append(f"Only {len(skills)} skills detected.")
        suggestions.append("List more technical and soft skills relevant to your target role.")
    else:
        weaknesses.append("Very few skills detected in the resume.")
        suggestions.append("Add a dedicated Skills section with relevant keywords.")

    # ── Action Verbs ──
    action_verbs = [
        "developed", "managed", "led", "created", "designed", "implemented",
        "improved", "increased", "decreased", "reduced", "achieved", "delivered",
        "built", "optimized", "architected", "coordinated",
    ]
    action_count = sum(1 for verb in action_verbs if verb in cleaned)
    if action_count >= 8:
        strengths.append(f"Good use of action verbs ({action_count} found) — makes impact clear.")
    elif action_count >= 4:
        strengths.append(f"Some action verbs used ({action_count} found).")
        suggestions.append("Use more strong action verbs (achieved, delivered, optimized, led) to describe your work.")
    else:
        suggestions.append("Start bullet points with strong action verbs (e.g., Developed, Led, Implemented).")

    # ── Quantifiable Results ──
    has_numbers = bool(re.search(r'\d+%|\d+x|\$\d+|\d+\s+(people|team|users|clients)', cleaned))
    if has_numbers:
        strengths.append("Includes quantifiable results — very effective!")
    else:
        suggestions.append("Add quantifiable results (e.g., 'Increased efficiency by 20%', 'Led a team of 10').")

    # ── Contact Info ──
    has_email = bool(re.search(r'\S+@\S+\.\S+', text))
    has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    if has_email and has_phone:
        strengths.append("Contact information (email & phone) is present.")
    else:
        suggestions.append("Ensure your email and phone number are clearly visible at the top.")

    # ── Summary / Objective ──
    if sections.get("summary"):
        strengths.append("Includes a professional summary or objective.")
    else:
        suggestions.append("Add a brief professional summary at the top to grab recruiters' attention.")

    # Deduplicate suggestions
    suggestions = list(dict.fromkeys(suggestions))
    weaknesses = list(dict.fromkeys(weaknesses))
    strengths = list(dict.fromkeys(strengths))

    return {
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "suggestions": suggestions[:7],
    }


def chat_with_resume(query: str, resume_text: str) -> str:
    """
    Answer user questions about the resume using context matching.

    Handles a wide variety of question types:
      - Skills, summary, projects, experience, education
      - Score, suggestions, strengths, weaknesses
      - Certifications, tools, technologies, frameworks
      - Companies, job titles, roles, dates
      - Achievements, languages, contact info
      - Role-specific questions (suitable roles, target jobs)
      - Generic context matching fallback
    """
    if not query or not resume_text:
        return "Please upload a resume first."

    query_lower = query.lower().strip()
    cleaned_text = _clean_text(resume_text)

    # ── META / GREETINGS ──
    greeting_words = {"hello", "hi", "hey", "greetings", "good morning",
                      "good afternoon", "good evening", "sup", "yo", "howdy"}
    if query_lower in greeting_words or any(g == query_lower.strip() for g in greeting_words):
        return (
            "👋 Hello! I'm your **Resume Assistant**. I can answer questions like:\n\n"
            "- *What skills are in my resume?*\n"
            "- *Summarize my resume*\n"
            "- *What projects do I have?*\n"
            "- *Tell me about my experience*\n"
            "- *What companies have I worked at?*\n"
            "- *What certifications do I have?*\n"
            "- *How can I improve?*\n"
            "- *What roles am I suitable for?*\n"
            "- *What is my resume score?*"
        )

    if "who are you" in query_lower or "what are you" in query_lower or "what can you" in query_lower:
        return (
            "I'm the **AI Resume Analyzer** 🤖. I analyze your resume PDF and can answer "
            "questions about your skills, experience, education, projects, and more. "
            "I can also give you a resume score, suggest improvements, and summarize your profile!"
        )

    if "thank" in query_lower or "thanks" in query_lower:
        return "You're welcome! 😊 Feel free to ask me anything else about your resume."

    # ── SKILLS / TECHNOLOGIES / TOOLS ──
    skill_patterns = {"skill", "technology", "technologies", "tech stack", "technical",
                      "programming", "language", "framework", "tool", "tools",
                      "know", "proficient", "expertise", "competencies"}
    if any(p in query_lower for p in skill_patterns):
        skills_list = extract_skills(resume_text)
        if not skills_list:
            return "I couldn't find many technical skills in your resume. Consider adding a 'Skills' section with relevant keywords."

        tech_skills = [s for s in skills_list if s not in (
            "leadership", "communication", "teamwork", "problem solving",
            "critical thinking", "project management", "agile", "scrum",
            "time management")]
        soft_skills = [s for s in skills_list if s in (
            "leadership", "communication", "teamwork", "problem solving",
            "critical thinking", "project management", "agile", "scrum",
            "time management")]

        response_parts = [f"**Skills Found ({len(skills_list)} total):**"]

        if tech_skills:
            response_parts.append(f"\n🛠️ **Technical:** {', '.join(tech_skills[:12])}"
                                  f"{' ...and more' if len(tech_skills) > 12 else ''}")
        if soft_skills:
            response_parts.append(f"\n🤝 **Soft Skills:** {', '.join(soft_skills)}")

        for skill in skills_list:
            if skill in query_lower:
                response_parts.append(f"\n✅ Yes, **{skill}** is in your resume!")

        specific_skills_asked = [w for w in query_lower.split()
                                 if w in SKILLS_DB and w not in skills_list]
        if specific_skills_asked:
            for s in specific_skills_asked[:3]:
                response_parts.append(f"\n❌ **{s}** was not detected. Consider adding it if relevant.")

        return "\n".join(response_parts)

    # ── STRENGTHS / WEAKNESSES ──
    if "strength" in query_lower:
        skills_list = extract_skills(resume_text)
        sug = generate_suggestions(resume_text, skills_list)
        if sug["strengths"]:
            return "**✅ Strengths of your resume:**\n\n" + "\n".join(f"• {s}" for s in sug["strengths"])
        return "I haven't identified specific strengths yet. Make sure you have sections for Experience, Education, and Skills."

    if "weakness" in query_lower or "weak" in query_lower or "gaps" in query_lower:
        skills_list = extract_skills(resume_text)
        sug = generate_suggestions(resume_text, skills_list)
        if sug["weaknesses"]:
            return "**⚠️ Areas for Improvement:**\n\n" + "\n".join(f"• {w}" for w in sug["weaknesses"])
        return "✨ No major weaknesses found! Your resume looks solid."

    # ── SUMMARY / OVERVIEW ──
    if "summary" in query_lower or "overview" in query_lower or "tell me about" in query_lower:
        return generate_summary(resume_text)

    # ── PROJECTS ──
    if "project" in query_lower:
        project_section = _extract_section_text(resume_text, "projects")
        if project_section and len(project_section) > 20:
            project_count = project_section.count("project") + project_section.count("built") + \
                            project_section.count("developed") + project_section.count("created")
            return (f"📁 **Projects ({max(1, project_count)} found):**\n\n"
                    f"{project_section[:600]}...")
        else:
            return "I couldn't find a dedicated Projects section. Add one to showcase your work!"

    # ── EXPERIENCE / WORK HISTORY ──
    if "experience" in query_lower or "work" in query_lower or "job" in query_lower or "career" in query_lower:
        exp_section = _extract_section_text(resume_text, "experience")
        if exp_section and len(exp_section) > 20:
            return f"💼 **Work Experience:**\n\n{exp_section[:600]}..."
        else:
            return "I couldn't find a clear Work Experience section. Make sure you list your job history."

    # ── COMPANIES / EMPLOYERS ──
    company_keywords = {"company", "companies", "employer", "employers", "organization",
                        "where", "worked at", "work at", "firm", "corp", "inc"}
    if any(k in query_lower for k in company_keywords):
        companies = re.findall(r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s*(?:Corp|Inc|LLC|Ltd|Technologies|Tech|Solutions|Systems|Group|Co|Company)',
                               resume_text)
        exp_section = _extract_section_text(resume_text, "experience")
        if companies:
            return "**🏢 Companies mentioned:**\n\n" + "\n".join(f"• {c}" for c in companies[:5])
        elif exp_section:
            lines = [l.strip() for l in exp_section.split('\n') if l.strip()]
            return f"Here is your work history — companies may be listed:\n\n{exp_section[:400]}..."
        else:
            return "I couldn't identify specific companies. Ensure your work experience includes employer names."

    # ── JOB TITLES / ROLES ──
    role_keywords = {"title", "role", "position", "designation", "job", "what do you do",
                     "current role", "current position"}
    if any(k in query_lower for k in role_keywords):
        titles = re.findall(r'(?:^|\n)\s*([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*)',
                           resume_text)
        meaningful_titles = [t for t in titles if len(t) > 5 and len(t) < 60
                             and "summary" not in t.lower()
                             and "experience" not in t.lower()
                             and "education" not in t.lower()
                             and "skills" not in t.lower()
                             and "projects" not in t.lower()]
        if meaningful_titles:
            return "**👔 Roles/Titles detected:**\n\n" + "\n".join(f"• {t}" for t in meaningful_titles[:5])
        return "I can't clearly identify job titles. Ensure they are prominently listed."

    # ── EDUCATION ──
    if "education" in query_lower or "degree" in query_lower or "university" in query_lower or "college" in query_lower or "school" in query_lower or "academic" in query_lower or "study" in query_lower or "graduate" in query_lower or "gpa" in query_lower:
        edu_section = _extract_section_text(resume_text, "education")
        if edu_section and len(edu_section) > 20:
            return f"🎓 **Education:**\n\n{edu_section[:500]}..."
        else:
            return "I couldn't find an Education section. Add your degrees and institutions."

    # ── CERTIFICATIONS ──
    cert_keywords = {"certification", "certificate", "certified", "license", "credential",
                     "pmp", "scrum", "aws certified", "azure certified"}
    if any(k in query_lower for k in cert_keywords) or any(c in cleaned_text for c in ["certif", "license"]):
        certs = re.findall(r'(?:certified|certification|certificate)[:\s]*([A-Za-z0-9\s\-]+)',
                           resume_text, re.IGNORECASE)
        if certs:
            return "**📜 Certifications detected:**\n\n" + "\n".join(f"• {c.strip()}" for c in certs[:5])
        skills_list = extract_skills(resume_text)
        cert_skills = [s for s in skills_list if 'certified' in s or 'pmp' in s or 'scrum master' in s]
        if cert_skills:
            return "**📜 Certifications detected:**\n\n" + "\n".join(f"• {s}" for s in cert_skills)
        return "No certifications detected in your resume. Add a Certifications section if you have any."

    # ── IMPROVEMENT / SUGGESTIONS ──
    improve_patterns = {"improve", "improvement", "suggestion", "suggestions",
                        "recommend", "recommendation", "better", "enhance",
                        "fix", "change", "update", "modify", "tip", "tips",
                        "advice", "help", "how to"}
    if any(p in query_lower for p in improve_patterns):
        skills_list = extract_skills(resume_text)
        suggestions_data = generate_suggestions(resume_text, skills_list)
        suggestions_text = "\n".join(f"• {s}" for s in suggestions_data["suggestions"][:7])
        response = "**🚀 Suggestions to improve your resume:**\n\n"
        if suggestions_data["strengths"]:
            response += "**✅ Strengths:**\n" + "\n".join(f"  - {s}" for s in suggestions_data["strengths"][:3]) + "\n\n"
        response += f"**💡 Recommendations:**\n{suggestions_text}"
        return response

    # ── SCORE / RATING ──
    score_patterns = {"score", "rate", "rating", "grade", "mark", "assessment",
                      "how good", "rank", "evaluate", "evaluation"}
    if any(p in query_lower for p in score_patterns):
        score_data = calculate_resume_score(resume_text)
        breakdown = score_data["breakdown"]
        breakdown_text = "\n".join(
            f"• **{k.replace('_', ' ').title()}:** {v}/{_get_max_score_helper(k)}"
            for k, v in breakdown.items()
        )
        return (
            f"**📊 Resume Score: {score_data['score']}/100 (Grade: {score_data['grade']})**\n\n"
            f"**Breakdown:**\n{breakdown_text}"
        )

    # ── ACHIEVEMENTS / ACCOMPLISHMENTS ──
    achievement_patterns = {"achievement", "accomplish", "award", "honor", "recognition",
                            "milestone", "result", "impact", "outstanding", "won",
                            "success", "promoted"}
    if any(p in query_lower for p in achievement_patterns):
        achievement_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', resume_text)
        for s in sentences:
            s_stripped = s.strip()
            if len(s_stripped) > 20:
                has_action = any(v in s_stripped.lower() for v in [
                    "achieved", "increased", "decreased", "improved", "delivered",
                    "reduced", "won", "awarded", "promoted", "led", "managed",
                    "generated", "launched", "exceeded", "optimized"
                ])
                has_number = bool(re.search(r'\d+%|\$\d+|\d+x|\d+\s+(people|team|user|client|customer)', s_stripped.lower()))
                if has_action or has_number:
                    achievement_sentences.append(s_stripped)
        if achievement_sentences:
            return "**🏆 Achievements detected:**\n\n" + "\n".join(f"• {s}" for s in achievement_sentences[:5])
        return "I didn't find clear achievements. Add quantifiable results (e.g., 'Increased sales by 30%')."

    # ── CONTACT INFO ──
    contact_patterns = {"contact", "email", "phone", "number", "address", "reach",
                        "linkedin", "github", "portfolio", "website", "profile"}
    if any(p in query_lower for p in contact_patterns):
        email = re.search(r'\S+@\S+\.\S+', resume_text)
        phone = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text)
        linkedin = re.search(r'(linkedin\.com[^\s]*)', resume_text, re.IGNORECASE)
        github = re.search(r'(github\.com[^\s]*)', resume_text, re.IGNORECASE)

        parts = []
        if email:
            parts.append(f"📧 Email: `{email.group()}`")
        if phone:
            parts.append(f"📞 Phone: `{phone.group()}`")
        if linkedin:
            parts.append(f"🔗 LinkedIn: {linkedin.group()}")
        if github:
            parts.append(f"💻 GitHub: {github.group()}")

        if parts:
            return "**📇 Contact Information:**\n\n" + "\n".join(parts)
        return "I couldn't find contact information. Add email and phone at the top of your resume."

    # ── LANGUAGES (SPOKEN) ──
    if "language" in query_lower and "programming" not in query_lower:
        spoken_languages = []
        lang_names = ["english", "spanish", "french", "german", "chinese", "japanese",
                      "korean", "italian", "portuguese", "russian", "arabic", "hindi",
                      "bilingual", "fluent", "native"]
        for lang in lang_names:
            if re.search(r'\b' + re.escape(lang) + r'\b', cleaned_text):
                spoken_languages.append(lang.title())
        if spoken_languages:
            return "**🗣️ Languages detected:**\n\n" + "\n".join(f"• {l}" for l in spoken_languages)
        return "I didn't detect spoken languages. Add a Languages section if relevant."

    # ── ROLE / JOB FIT QUESTIONS ──
    role_fit_patterns = {"suitable", "fit for", "role", "position", "job", "career",
                         "what kind of", "what type of", "should i apply",
                         "good for", "qualify", "qualified"}
    if any(p in query_lower for p in role_fit_patterns) and any(
            w in query_lower for w in ["role", "position", "job", "fit", "suitable"]):
        skills_list = extract_skills(resume_text)
        if skills_list:
            role_suggestions = []
            if any(s in skills_list for s in ["python", "java", "javascript", "c++"]):
                role_suggestions.append("Software Engineer / Developer")
            if any(s in skills_list for s in ["react", "angular", "vue", "html", "css"]):
                role_suggestions.append("Frontend Developer")
            if any(s in skills_list for s in ["node.js", "django", "flask", "express"]):
                role_suggestions.append("Backend Developer")
            if any(s in skills_list for s in ["machine learning", "deep learning", "tensorflow", "pytorch"]):
                role_suggestions.append("Machine Learning Engineer / Data Scientist")
            if any(s in skills_list for s in ["aws", "azure", "gcp", "docker", "kubernetes"]):
                role_suggestions.append("DevOps Engineer / Cloud Architect")
            if any(s in skills_list for s in ["sql", "mysql", "postgresql", "mongodb"]):
                role_suggestions.append("Data Analyst / Database Administrator")
            if any(s in skills_list for s in ["project management", "agile", "scrum", "leadership"]):
                role_suggestions.append("Project Manager / Team Lead")

            if role_suggestions:
                return ("**🎯 Based on your skills, you might be a good fit for:**\n\n"
                        + "\n".join(f"• {r}" for r in role_suggestions))
            return "I can see your skills, but I need more context to suggest specific roles."

    # ── DATES / TIMELINE ──
    date_patterns = {"date", "year", "month", "timeline", "duration", "how long",
                     "when", "time", "period", "start", "end", "present"}
    if any(p in query_lower for p in date_patterns):
        dates_found = re.findall(r'\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:Present|(?:19|20)\d{2}|[A-Z][a-z]+ \d{4}))?', resume_text)
        if dates_found:
            return "**📅 Dates / Timeline found:**\n\n" + "\n".join(f"• {d}" for d in dates_found[:8])
        return "I couldn't find clear dates. Ensure your experience includes time periods."

    # ── AI-POWERED TEXT GENERATION (FLAN-T5) ──
    text_gen = _get_text_gen()
    if text_gen:
        try:
            context = resume_text[:1500]
            prompt = f"Answer based on resume.\n\nResume: {context}\n\nQuestion: {query}\nAnswer:"
            result = text_gen(prompt, max_length=200, do_sample=False)
            if result and len(result) > 0:
                answer = result[0]['generated_text'].strip()
                if answer and len(answer) > 5:
                    return answer
        except Exception:
            pass

    # ── SMART SENTENCE RETRIEVAL ──
    sentences = re.split(r'(?<=[.!?])\s+', resume_text)
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    stop_words = {"the", "and", "for", "are", "was", "were", "has", "have", "had",
                  "but", "not", "you", "all", "can", "its", "how", "why", "what",
                  "when", "where", "who", "did", "get", "got", "use", "used",
                  "also", "than", "then", "that", "this", "with", "from", "they",
                  "your", "about", "into", "over", "after", "before", "just",
                  "some", "them", "these", "those", "much", "more", "very"}
    query_words = query_words - stop_words

    scored_sentences = []
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 15:
            continue
        s_lower = s.lower()
        words_in_s = set(s_lower.split()) - stop_words
        if not words_in_s:
            continue
        common = query_words & words_in_s
        if common:
            score = len(common) * 2 + (1 if any(v in s_lower for v in query_lower.split() if len(v) > 3) else 0)
            scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    if scored_sentences and scored_sentences[0][0] >= 2:
        top = scored_sentences[:3]
        answer = " ".join(s for _, s in top)
        if len(answer) > 700:
            answer = answer[:697] + "..."
        return f"**Based on your resume:**\n\n{answer}"
    else:
        return (
            "💬 I can answer many questions about your resume! Try asking:\n\n"
            "📌 **Skills** — *'What skills are in my resume?'*\n"
            "📌 **Summary** — *'Summarize my resume'*\n"
            "📌 **Projects** — *'What projects do I have?'*\n"
            "📌 **Experience** — *'Tell me about my experience'*\n"
            "📌 **Companies** — *'Where have I worked?'*\n"
            "📌 **Education** — *'What is my education background?'*\n"
            "📌 **Certifications** — *'What certifications do I have?'*\n"
            "📌 **Score** — *'What is my resume score?'*\n"
            "📌 **Suggestions** — *'How can I improve?'*\n"
            "📌 **Strengths** — *'What are my strengths?'*\n"
            "📌 **Roles** — *'What roles am I suitable for?'*\n"
            "📌 **Contact** — *'What is my contact info?'*\n"
            "📌 **Achievements** — *'What achievements do I have?'*\n"
            "📌 **Dates** — *'What are my employment dates?'*\n"
            "📌 **Weaknesses** — *'What are my weaknesses?'*"
        )


def _get_max_score_helper(key: str) -> int:
    """Return max possible score for a breakdown category (duplicated for import-free use)."""
    return {"length": 30, "sections": 25, "skills": 25, "action_verbs": 20}.get(key, 25)

