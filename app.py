# app.py
import os
import re
import json
import html
from datetime import datetime
import requests
import streamlit as st

# --- Azure Document Intelligence SDK (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight — Advanced", page_icon="📚", layout="wide")
st.title("📚 Suvichaar — Literature Insight (Advanced)")
st.caption("Upload/paste text → Auto-detect category → Category template → Generate detailed study data + Q&As + Quiz → Edit → Export.")

# =========================
# SECRETS / CONFIG
# =========================
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_VERSION):
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml`: AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION.")

# =========================
# OCR (IMAGES / PDFs)
# =========================
def ocr_read_any(bytes_blob: bytes) -> str:
    """Use Azure DI 'prebuilt-read' to extract text from images or PDFs (optional)."""
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return ""
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return ""
    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT.rstrip("/"),
            credential=AzureKeyCredential(AZURE_DI_KEY)
        )
        poller = client.begin_analyze_document("prebuilt-read", body=bytes_blob)
        doc = poller.result()
        parts = []
        if getattr(doc, "pages", None):
            for p in doc.pages:
                lines = [ln.content for ln in getattr(p, "lines", []) or [] if getattr(ln, "content", None)]
                page_txt = "\n".join(lines).strip()
                if page_txt:
                    parts.append(page_txt)
        elif getattr(doc, "paragraphs", None):
            parts.append("\n".join(pp.content for pp in doc.paragraphs if getattr(pp, "content", None)))
        else:
            raw = (getattr(doc, "content", "") or "").strip()
            if raw:
                parts.append(raw)
        return "\n".join(parts).strip()
    except Exception:
        return ""

# =========================
# SAFE SANITIZATION WRAPPER
# =========================
def make_classroom_safe(text: str) -> str:
    """Replace potentially sensitive words with classroom-friendly phrasing to reduce filter blocks."""
    replacements = {
        r"\bkill(ed|ing)?\b": "harm",
        r"\bmurder(ed|ing)?\b": "serious harm",
        r"\bdeath\b": "loss",
        r"\bdie(s|d)?\b": "pass away",
        r"\bblood\b": "red liquid",
        r"\bsuicide\b": "personal struggle",
        r"\bviolence\b": "conflict",
        r"\bhate(d|s)?\b": "dislike",
        r"\babuse(d|s|ive)?\b": "mistreatment",
        r"\bdrugs?\b": "substances",
        r"\balcohol\b": "drinks",
        r"\bsex(ual|ually)?\b|\bsex\b": "personal topic"
    }
    for pat, sub in replacements.items():
        text = re.sub(pat, sub, text, flags=re.IGNORECASE)
    return text

# =========================
# AZURE GPT CALL
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=7000, force_json=True):
    """Calls Azure OpenAI Chat Completions. response_format=json_object forces valid JSON."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return True, content
        # content filter hint
        if r.status_code == 400 and "filtered" in r.text.lower():
            return False, "FILTERED"
        return False, f"Azure error {r.status_code}: {r.text[:800]}"
    except Exception as e:
        return False, f"Azure request failed: {e}"

def robust_parse(s: str):
    """Attempts normal parse → extract inner {...} → graceful failure."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

# =========================
# LANGUAGE DETECTION
# =========================
def detect_hi_or_en(text: str) -> str:
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total = devanagari + latin
    if total == 0:
        return "en"
    return "hi" if (devanagari / total) >= 0.25 else "en"

# =========================
# CATEGORY DETECTION
# =========================
CATEGORIES = [
    "poetry", "play", "story", "essay", "biography", "autobiography",
    "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"
]

def heuristic_guess_category(txt: str) -> str:
    t = txt.strip()
    lower = t.lower()
    if re.search(r'^\s*[A-Z].+\n[A-Z].+', t) and re.search(r'\n[A-Z][a-z]+:', t):  # NAME: dialogue
        return "play"
    if re.search(r'^[^\n]{0,80}\n[^\n]{0,80}\n[^\n]{0,80}(\n|$)', t) and re.search(r'[,.!?]\s*$', t) is None:
        if len([ln for ln in t.splitlines() if ln.strip()]) <= 20:
            return "poetry"
    if any(k in lower for k in ["dear sir", "dear madam", "yours faithfully", "yours sincerely", "to,", "from:", "subject:"]):
        return "letter"
    if any(k in lower for k in ["dear diary", "today i", "date:", "entry"]):
        return "diary"
    if any(k in lower for k in ["once upon a time", "there was", "moral", "village"]):
        return "folk_tale"
    if any(k in lower for k in ["according to", "in conclusion", "therefore", "on the other hand"]):
        return "essay"
    if any(k in lower for k in ["i was born", "my childhood", "in my life"]) and " i " in lower:
        return "autobiography"
    if any(k in lower for k in ["he was born", "she was born", "in year", "died in", "awarded"]):
        return "biography"
    if any(k in lower for k in ["ladies and gentlemen", "audience", "i stand before you", "speech"]):
        return "speech"
    if any(k in lower for k in ["report on", "submitted to", "findings", "method", "observation"]):
        return "report"
    return "story"  # default

def classify_with_gpt(txt: str, lang: str) -> str:
    """Ask Azure to classify; fall back to heuristic."""
    safe = make_classroom_safe(txt)
    sys = (
        "You are a precise classifier for school literature. "
        "Return ONLY one lowercase label from this set: poetry, play, story, essay, biography, autobiography, "
        "speech, letter, diary, report, folk_tale, myth, legend."
    )
    user = f"Text:\n{safe}\n\nLabel only."
    ok, out = call_azure_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=24, force_json=False
    )
    if not ok or not out:
        return heuristic_guess_category(txt)
    label = out.strip().lower()
    label = re.sub(r'[^a-z_]', '', label)
    return label if label in CATEGORIES else heuristic_guess_category(txt)

# =========================
# CATEGORY TEMPLATES (per-category core + extras)
# =========================
UNIVERSAL_SECTIONS = [
    "Title & Creator (Author/Poet/Speaker)",
    "Introduction / Context",
    "Central Idea / Gist",
    "Summary (section-wise)",
    "Themes & Messages",
    "Tone & Mood",
    "Important Quotes / Key Lines",
    "Vocabulary & Meanings",
    "Background of Author/Poet/Speaker",
    "Question Bank (Short/Long/MCQ)",
    "Classroom Activities / Practice"
]

def make_study_template(category: str) -> dict:
    cat = (category or "").lower()
    tpl = {"category": cat, "sections": list(UNIVERSAL_SECTIONS), "extras": []}

    if cat == "poetry":
        tpl["sections"].insert(5, "Structure (stanzas, rhyme scheme, meter)")
        tpl["extras"] = ["Speaker/Voice", "Literary & Sound Devices", "Imagery & Symbolism", "Emotional Arc"]
    elif cat == "play":
        tpl["sections"].insert(3, "Scene/Act-wise Summary")
        tpl["extras"] = ["Characters & Relationships", "Setting & Stage Directions", "Dialogue Beats", "Dramatic Devices (soliloquy, irony)", "Plot Points & Conflict"]
    elif cat == "story":
        tpl["extras"] = ["Narrative Voice", "Characters", "Setting", "Plot Points (exposition→resolution)", "Conflict"]
    elif cat == "essay":
        tpl["extras"] = ["Thesis/Main Claim", "Key Points with Evidence", "Structure (intro/body/conclusion)", "Rhetorical Devices"]
    elif cat == "biography":
        tpl["extras"] = ["Subject Timeline", "Qualities/Values", "Key Incidents", "Impact/Contributions"]
    elif cat == "autobiography":
        tpl["extras"] = ["Episodes & Reflections", "Voice & Style", "Lessons Learnt"]
    elif cat == "speech":
        tpl["extras"] = ["Audience & Purpose", "Key Points", "Rhetorical Devices", "Call to Action"]
    elif cat == "letter":
        tpl["extras"] = ["Type (formal/informal)", "Salutation/Closing", "Body Points", "Tone & Register"]
    elif cat == "diary":
        tpl["extras"] = ["Date/Time Hint", "Events", "Feelings & Reflection"]
    elif cat == "report":
        tpl["extras"] = ["Topic", "Sections (Intro/Method/Observation/Discussion/Conclusion)", "Findings", "Recommendations"]
    elif cat in ("folk_tale", "myth", "legend"):
        tpl["extras"] = ["Characters", "Setting", "Plot Outline", "Motifs/Symbols", "Moral or Cultural Significance"]

    return tpl

# =========================
# DATA SCHEMA FOR MODEL
# =========================
ABOUT_AUTHOR = {
    "name": "",
    "era_or_period": "",
    "nationality_or_region": "",
    "notable_works": [],
    "themes_or_motifs": [],
    "influences_or_style": "",
    "relevance_to_text": ""
}

CHARACTER_SKETCH = [{
    "name": "",
    "role": "",
    "traits": [],
    "motives": [],
    "arc_or_change": "",
    "relationships": [{"with": "", "nature": "", "note": ""}],
    "key_quotes": [{"quote": "", "explanation": ""}]
}]

THEME_BLOCK = [{"theme": "", "explanation": "", "evidence_quotes": []}]

ACTIVITIES_BLOCK = {
    "pre_reading": [{"title": "", "steps": [], "duration_min": 10}],
    "during_reading": [{"title": "", "steps": [], "strategy": ""}],
    "post_reading": [{"title": "", "steps": [], "outcome": ""}],
    "creative_tasks": [{"title": "", "type": "", "prompt": ""}],
    "projects": [{"title": "", "deliverable": "", "criteria": []}]
}

ASSESSMENT_RUBRIC = [{
    "criterion": "",
    "levels": {"exemplary": "", "proficient": "", "developing": "", "emerging": ""}
}]

BASE_SAFE_FIELDS = {
    "language": "en|hi",
    "text_type": "category label",
    "literal_meaning": "",
    "figurative_meaning": "",
    "tone_mood": "",
    "one_sentence_takeaway": "",
    "executive_summary": "",
    "inspiration_hook": "",
    "why_it_matters": "",
    "study_tips": [],
    "extension_reading": [],
    "emotional_arc": [{"beat": "", "feeling": "", "evidence": ""}],
    "questions_short": [{"q": "", "a": ""}],
    "questions_long": [{"q": "", "a": ""}]
}

SCHEMAS = {
    "poetry": {
        **BASE_SAFE_FIELDS,
        "speaker_or_voice": "",
        "structure_overview": {"stanzas": "", "approx_line_count": "", "rhyme_scheme": "", "meter_or_rhythm": ""},
        "themes_detailed": THEME_BLOCK,
        "devices": [{"name": "", "evidence": "", "explanation": ""}],
        "imagery_map": [{"sense": "", "evidence": "", "effect": ""}],
        "symbol_table": [{"symbol": "", "meaning": "", "evidence": ""}],
        "line_by_line": [{"line": "", "explanation": "", "device_notes": ""}],
        "context_or_background": "",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": [],
        "quote_bank": [],
        "comparative_texts": [{"title": "", "note": ""}],
        "cross_curricular_links": [{"subject": "", "idea": ""}],
        "adaptation_ideas": [],
        "vocabulary_glossary": [{"term": "", "meaning": ""}],
        "misconceptions": []
    },
    "play": {
        **BASE_SAFE_FIELDS,
        "characters": CHARACTER_SKETCH,
        "setting": "",
        "conflict": "",
        "dialogue_beats": [{"speaker": "", "line": "", "note": ""}],
        "stage_directions": "",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": [],
        "quote_bank": [],
        "comparative_texts": [{"title": "", "note": ""}],
        "cross_curricular_links": [{"subject": "", "idea": ""}],
        "adaptation_ideas": []
    },
    "story": {
        **BASE_SAFE_FIELDS,
        "narrative_voice": "",
        "setting": "",
        "characters": CHARACTER_SKETCH,
        "plot_points": [{"stage": "", "what_happens": "", "evidence": ""}],
        "conflict": "",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": [],
        "quote_bank": [],
        "comparative_texts": [{"title": "", "note": ""}],
        "cross_curricular_links": [{"subject": "", "idea": ""}],
        "adaptation_ideas": []
    },
    "essay": {
        **BASE_SAFE_FIELDS,
        "thesis": "",
        "key_points": [{"point": "", "evidence_or_example": "", "counterpoint_if_any": ""}],
        "structure": "",
        "tone_register": "",
        "rhetorical_devices": [{"name": "", "evidence": "", "effect": ""}],
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": [],
        "quote_bank": []
    },
    "biography": {
        **BASE_SAFE_FIELDS,
        "subject": "",
        "timeline": [{"year_or_age": "", "event": "", "impact": ""}],
        "qualities": [],
        "influence_or_impact": "",
        "notable_works_or_contributions": [],
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "autobiography": {
        **BASE_SAFE_FIELDS,
        "author": "",
        "episodes": [{"when": "", "event": "", "reflection": "", "lesson": ""}],
        "themes_detailed": THEME_BLOCK,
        "voice_and_style": "",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "speech": {
        **BASE_SAFE_FIELDS,
        "audience": "",
        "purpose": "",
        "key_points": [],
        "rhetorical_devices": [{"name": "", "evidence": "", "effect": ""}],
        "call_to_action": "",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "letter": {
        **BASE_SAFE_FIELDS,
        "letter_type": "",
        "salutation": "",
        "body_points": [{"point": "", "example_or_reason": ""}],
        "closing": "",
        "tone_register": "",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "diary": {
        **BASE_SAFE_FIELDS,
        "date_or_time_hint": "",
        "events": [],
        "feelings": "",
        "reflection": "",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "report": {
        **BASE_SAFE_FIELDS,
        "topic": "",
        "sections": [{"heading": "", "summary": ""}],
        "findings": [],
        "recommendations": [],
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "folk_tale": {
        **BASE_SAFE_FIELDS,
        "characters": CHARACTER_SKETCH,
        "setting": "",
        "plot_outline": [],
        "repeating_patterns_or_motifs": [],
        "moral_or_lesson": "",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "myth": {
        **BASE_SAFE_FIELDS,
        "deities_or_symbols": [],
        "origin_or_explanation": "",
        "plot_outline": [],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "legend": {
        **BASE_SAFE_FIELDS,
        "hero_or_figure": "",
        "historical_backdrop": "",
        "notable_events": [],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    }
}

# =========================
# PROMPT BUILDER (DATA + QAs + QUIZ)
# =========================
def build_prompt(category: str, language_code: str, max_per_list: int, q_short_n: int, q_long_n: int, quiz_n: int) -> str:
    schema = SCHEMAS.get(category, SCHEMAS["story"])

    rules = f"""
Return ONLY JSON with this exact top-level shape:
{{
  "template_used": "{category}",
  "data": {{ ... }},
  "qas": {{
    "short": [{{"q":"...","a":"..."}}],   // up to {q_short_n}
    "long":  [{{"q":"...","a":"..."}}]    // up to {q_long_n}
  }},
  "quiz": [{{"q":"...","choices":["A","B","C","D"],"answer":"A"}}]  // up to {quiz_n}
}}

HARD RULES:
- Values must be in the target explanation language ("{language_code}").
- For each list in "data", include at most {max_per_list} items (concise but specific).
- For "qas.short", generate up to {q_short_n} items. For "qas.long", up to {q_long_n}.
- For "quiz", generate up to {quiz_n} MCQs with one correct answer.
- Use classroom-safe, neutral phrasing.
- Do not include any text outside JSON. No markdown code fences.
- If the text does not support a section, omit that section entirely.
""".strip()

    return (
        f"You are an expert literature teacher. Fill study DATA for a '{category}' text.\n"
        f"Target language: {language_code}\n"
        + rules
        + "\n\nDATA SCHEMA (shape for the 'data' object):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )

# =========================
# NORMALIZERS (prevent .get() crashes, clean shapes)
# =========================
def _normalize_list(val):
    return val if isinstance(val, list) else ([] if val in (None, "") else [val])

def norm_q_short(x):
    if isinstance(x, dict):
        return {"q": str(x.get("q","")).strip(), "a": str(x.get("a","")).strip()}
    return {"q": str(x).strip(), "a": ""}

def norm_q_long(x):
    if isinstance(x, dict):
        return {"q": str(x.get("q","")).strip(), "a": str(x.get("a","")).strip()}
    return {"q": str(x).strip(), "a": ""}

def norm_quiz(x):
    if isinstance(x, dict):
        q = str(x.get("q","")).strip()
        choices = x.get("choices", [])
        if not isinstance(choices, list):
            choices = [str(choices)]
        choices = [str(c) for c in choices]
        answer = str(x.get("answer","")).strip()
        return {"q": q, "choices": choices, "answer": answer}
    return {"q": str(x).strip(), "choices": [], "answer": ""}

def normalize_data(data: dict) -> dict:
    """Soft-normalize common sections so rendering is safe."""
    if not isinstance(data, dict):
        return {}

    # devices always list of dicts
    if "devices" in data:
        devs = _normalize_list(data.get("devices"))
        out = []
        for d in devs:
            if isinstance(d, dict):
                out.append({"name": d.get("name",""), "evidence": d.get("evidence",""), "explanation": d.get("explanation","")})
        data["devices"] = out

    # characters
    if "characters" in data:
        chars = _normalize_list(data.get("characters"))
        c_out = []
        for c in chars:
            if isinstance(c, dict):
                c_out.append({
                    "name": c.get("name",""),
                    "role": c.get("role",""),
                    "traits": c.get("traits", []) if isinstance(c.get("traits"), list) else [],
                    "motives": c.get("motives", []) if isinstance(c.get("motives"), list) else [],
                    "arc_or_change": c.get("arc_or_change",""),
                    "key_quotes": c.get("key_quotes", []) if isinstance(c.get("key_quotes"), list) else []
                })
        data["characters"] = c_out

    # themes_detailed
    if "themes_detailed" in data:
        th = _normalize_list(data.get("themes_detailed"))
        t_out = []
        for t in th:
            if isinstance(t, dict):
                ev = t.get("evidence_quotes", [])
                if not isinstance(ev, list):
                    ev = [str(ev)]
                t_out.append({"theme": t.get("theme",""), "explanation": t.get("explanation",""), "evidence_quotes": [str(x) for x in ev]})
        data["themes_detailed"] = t_out

    # imagery_map / symbol_table / plot_points / dialogue_beats / line_by_line et al — ensure lists
    for key in ["imagery_map", "symbol_table", "plot_points", "dialogue_beats", "line_by_line", "vocabulary_glossary",
                "quote_bank", "comparative_texts", "cross_curricular_links", "adaptation_ideas", "homework"]:
        if key in data:
            lst = _normalize_list(data.get(key))
            # keep only dicts for complex tables; strings for quote_bank ok
            if key in ["quote_bank", "adaptation_ideas", "homework"]:
                data[key] = [str(x) for x in lst if isinstance(x, (str, int, float))]
            else:
                data[key] = [x for x in lst if isinstance(x, dict)]

    return data

def normalize_qas_quiz(qas: dict, quiz: list):
    # QAs
    if not isinstance(qas, dict):
        qas = {}
    qas["short"] = [norm_q_short(x) for x in _normalize_list(qas.get("short", []))]
    qas["long"]  = [norm_q_long(x)  for x in _normalize_list(qas.get("long",  []))]

    # Quiz
    quiz = [norm_quiz(x) for x in _normalize_list(quiz)]
    return qas, quiz

# =========================
# HTML EXPORT
# =========================
def list_html(items):
    if not items:
        return "<p>—</p>"
    lis = "".join(f"<li>{html.escape(str(i))}</li>" for i in items)
    return f"<ul>{lis}</ul>"

def build_portable_html(bundle: dict) -> str:
    cat = bundle.get("category","").title()
    data = bundle.get("data",{}) or {}
    qas  = bundle.get("qas",{}) or {}
    quiz = bundle.get("quiz",[]) or []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def g(key, default="—"):
        v = data.get(key)
        return html.escape(v) if isinstance(v, str) and v.strip() else default

    # Themes rows
    theme_rows = ""
    for t in data.get("themes_detailed", []) or []:
        theme = html.escape(str(t.get("theme","")))
        expl  = html.escape(str(t.get("explanation","")))
        ev    = " | ".join([html.escape(str(x)) for x in (t.get("evidence_quotes",[]) or [])])
        theme_rows += f"<tr><td>{theme}</td><td>{expl}</td><td>{ev}</td></tr>"

    # Q&A lists (render as bullet lines)
    short_items = [f"Q: {s.get('q','')} — A: {s.get('a','')}" for s in (qas.get("short") or [])]
    long_items  = [f"Q: {l.get('q','')} — A: {l.get('a','')}"  for l in (qas.get("long")  or [])]

    # Quiz rows
    quiz_rows = ""
    for q in quiz:
        qq = html.escape(q.get("q",""))
        ch = q.get("choices",[])
        ch_txt = " | ".join([html.escape(str(c)) for c in ch]) if isinstance(ch, list) else html.escape(str(ch))
        ans = html.escape(q.get("answer",""))
        quiz_rows += f"<tr><td>{qq}</td><td>{ch_txt}</td><td>{ans}</td></tr>"

    html_doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>'
        f"<title>Literature Insight — {html.escape(cat)}</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;line-height:1.5;margin:24px;color:#0f172a}"
        "h1,h2{margin:0.4em 0}"
        ".card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;background:#fff}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}"
        ".small{color:#475569;font-size:12px}"
        "</style></head><body>"
        f"<h1>Literature Insight — {html.escape(cat)}</h1>"
        f'<div class="small">Generated: {html.escape(ts)}</div>'

        f'<div class="card"><h2>Executive Summary</h2><p>{g("executive_summary")}</p></div>'
        f'<div class="card"><h2>Inspiration Hook</h2><p>{g("inspiration_hook")}</p></div>'
        f'<div class="card"><h2>Why It Matters</h2><p>{g("why_it_matters")}</p></div>'

        f'<div class="card"><h2>Study Tips</h2>{list_html(data.get("study_tips"))}</div>'
        f'<div class="card"><h2>Quote Bank</h2>{list_html(data.get("quote_bank"))}</div>'

        '<div class="card"><h2>Themes & Evidence</h2>'
        '<table><thead><tr><th>Theme</th><th>Explanation</th><th>Evidence</th></tr></thead>'
        f"<tbody>{theme_rows or '<tr><td colspan=\"3\">—</td></tr>'}</tbody></table></div>"

        f'<div class="card"><h2>Short Q&A</h2>{list_html(short_items)}</div>'
        f'<div class="card"><h2>Long Q&A</h2>{list_html(long_items)}</div>'

        '<div class="card"><h2>Quiz (MCQ)</h2>'
        '<table><thead><tr><th>Question</th><th>Choices</th><th>Answer</th></tr></thead>'
        f"<tbody>{quiz_rows or '<tr><td colspan=\"3\">—</td></tr>'}</tbody></table></div>"

        '<div class="small">© Suvichaar Literature Insight</div>'
        "</body></html>"
    )
    return html_doc

# =========================
# UI INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a poem/play/story/essay (optional)", height=160, placeholder="e.g., Whose woods these are I think I know…")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols_top = st.columns(6)
with cols_top[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols_top[1]:
    detail_level = st.slider("Detail level", 1, 5, 5, help="Higher = richer lists and explanations.")
with cols_top[2]:
    max_per_list = st.slider("Max items per data list", 1, 8, 5)
with cols_top[3]:
    q_short_n = st.slider("Short Q&A (count)", 1, 10, 6)
with cols_top[4]:
    q_long_n = st.slider("Long Q&A (count)", 1, 10, 6)
with cols_top[5]:
    quiz_n = st.slider("Quiz (MCQs)", 1, 10, 10)

st.markdown("#### Display controls")
dcols = st.columns(7)
with dcols[0]:
    show_author = st.toggle("Author", value=True)
with dcols[1]:
    show_characters = st.toggle("Characters", value=True)
with dcols[2]:
    show_line_by_line = st.toggle("Line/Beats", value=True)
with dcols[3]:
    show_devices_table = st.toggle("Devices", value=True)
with dcols[4]:
    show_activities = st.toggle("Activities", value=True)
with dcols[5]:
    show_rubric = st.toggle("Rubric", value=True)
with dcols[6]:
    allow_padding = st.toggle("Auto-pad short lists", value=False, help="If on, missing items are padded with safe placeholders.")

run = st.button("🔎 Analyze & Generate")

# =========================
# MAIN
# =========================
if run:
    # Build source text
    source_text = (text_input or "").strip()
    if files and not source_text:
        with st.spinner("Running OCR on uploaded file…"):
            blob = files.read()
            ocr_text = ocr_read_any(blob)
            if ocr_text:
                source_text = ocr_text
                st.success("OCR text extracted:")
                with st.expander("Show OCR text"):
                    st.write(ocr_text[:20000])
            else:
                st.error("OCR returned no text. Try a clearer image or paste the text manually.")
                st.stop()

    if not source_text:
        st.error("Please paste text or upload a file.")
        st.stop()

    # Language
    detected = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else "hi" if lang_choice == "Hindi" else detected

    # Category detection
    guessed = heuristic_guess_category(source_text)
    try:
        cat = classify_with_gpt(source_text, explain_lang) or guessed
    except Exception:
        cat = guessed

    st.markdown("#### Detected")
    st.markdown(f"- **Language:** `{explain_lang}`  \n- **Category:** `{cat}`")

    # Category Template
    study_template = make_study_template(cat)

    # Build prompt and call model (DATA + QAs + QUIZ)
    safe_text = make_classroom_safe(source_text)
    system_msg = (
        "You are a veteran literature teacher for school students. "
        "Analyze the text in a CLASSROOM-SAFE way. Avoid explicit/graphic language. "
        "Stick to evidence from the text."
        + (" Respond in Hindi." if explain_lang.startswith("hi") else " Respond in English.")
    )
    user_msg = (
        f"TEXT TO ANALYZE (verbatim):\n{safe_text}\n\n"
        + build_prompt(cat, explain_lang, max_per_list, q_short_n, q_long_n, quiz_n)
    )

    with st.spinner("Calling Azure to generate structured data + Q&As + quiz…"):
        ok, content = call_azure_chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2 if detail_level >= 4 else 0.12,
            max_tokens=7000,
            force_json=True
        )

    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in tighter student-safe mode…")
        ok, content = call_azure_chat(
            [{"role":"system","content":"You are a cautious school literature teacher. Avoid explicit terms; use neutral wording."},
             {"role":"user","content":user_msg}],
            temperature=0.0, max_tokens=6000, force_json=True
        )

    if not ok:
        st.error(content)
        st.stop()

    parsed = robust_parse(content)
    data = (parsed.get("data") or {}) if isinstance(parsed, dict) else {}
    qas  = (parsed.get("qas")  or {}) if isinstance(parsed, dict) else {}
    quiz = (parsed.get("quiz") or []) if isinstance(parsed, dict) else []

    # Normalize shapes to avoid .get() errors
    data = normalize_data(data)
    qas, quiz = normalize_qas_quiz(qas, quiz)

    # Optional padding to match requested counts (safe placeholders)
    if allow_padding:
        def pad(arr, n, make):
            arr = arr or []
            while len(arr) < n:
                arr.append(make(len(arr)+1))
            return arr

        # themes_detailed
        if "themes_detailed" in data:
            data["themes_detailed"] = pad(
                data["themes_detailed"], max_per_list,
                lambda i: {"theme": f"Additional theme {i}", "explanation": "Reinforced insight.", "evidence_quotes": []}
            )
        # quote_bank
        if "quote_bank" in data:
            data["quote_bank"] = pad(data["quote_bank"], max_per_list, lambda i: f"Reinforced quote {i}.")

        # QAs & Quiz
        qas["short"] = pad(qas.get("short", []), q_short_n, lambda i: {"q": f"Additional short question {i}", "a": "Concise answer (reinforced)."})
        qas["long"]  = pad(qas.get("long",  []), q_long_n,  lambda i: {"q": f"Additional analytical question {i}", "a": "Evidence-based model answer (reinforced)."})
        quiz         = pad(quiz, quiz_n, lambda i: {"q": f"Extra MCQ {i}", "choices": ["A","B","C","D"], "answer":"A"})

    # ============= PRESENTATION LAYOUT =============
    tabs = st.tabs([
        "Study Template", "Overview", "Text Insights", "Author & Characters",
        "Themes & Quotes", "Activities & Rubrics", "Q&A", "Quiz", "Export"
    ])

    # ----- Study Template -----
    with tabs[0]:
        st.subheader("🧩 Category-specific Study Template")
        st.write(f"**Category:** {study_template['category'].title()}")
        st.markdown("**Core Sections**")
        st.write("\n".join(f"• {s}" for s in study_template["sections"]))
        if study_template.get("extras"):
            st.markdown("**Extras (focus for this category)**")
            st.write("\n".join(f"• {s}" for s in study_template["extras"]))

    # ----- Overview -----
    with tabs[1]:
        st.subheader("🧭 Executive Summary")
        st.write(data.get("executive_summary") or data.get("one_sentence_takeaway") or "—")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✨ Inspiration Hook")
            st.write(data.get("inspiration_hook") or "—")
        with c2:
            st.subheader("🎯 Why It Matters")
            st.write(data.get("why_it_matters") or "—")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📝 Study Tips")
            st.write("\n".join(f"• {x}" for x in (data.get("study_tips") or [])) or "—")
        with c4:
            st.subheader("📚 Extension Reading")
            st.write("\n".join(f"• {x}" for x in (data.get("extension_reading") or [])) or "—")

    # ----- Text Insights -----
    with tabs[2]:
        lcol, rcol = st.columns(2)
        with lcol:
            st.subheader("📘 Literal meaning")
            st.write(data.get("literal_meaning","—"))
            st.subheader("🌊 Figurative meaning / themes")
            st.write(data.get("figurative_meaning","—"))
        with rcol:
            st.subheader("🎼 Tone & Mood")
            st.write(data.get("tone_mood","—"))
            st.subheader("✅ One-sentence takeaway")
            st.write(data.get("one_sentence_takeaway","—"))

        if cat == "poetry":
            if data.get("structure_overview"):
                st.subheader("🧩 Structure overview")
                st.json(data["structure_overview"], expanded=False)
            if show_line_by_line and data.get("line_by_line"):
                st.subheader("📖 Line-by-line")
                # Show only if meaningful items exist
                rows = []
                for it in data["line_by_line"]:
                    if isinstance(it, dict) and any(it.get(k) for k in ("line","explanation","device_notes")):
                        rows.append({"line": it.get("line",""), "explanation": it.get("explanation",""), "device_notes": it.get("device_notes","")})
                if rows:
                    st.table(rows)
                else:
                    st.write("—")
            if show_devices_table and (data.get("devices") or []):
                st.subheader("🎭 Devices")
                rows = []
                for d in data.get("devices") or []:
                    if isinstance(d, dict):
                        rows.append({"device": d.get("name",""), "evidence": d.get("evidence",""), "why": d.get("explanation","")})
                st.table(rows or [{"device":"—","evidence":"—","why":"—"}])
            if data.get("imagery_map"):
                st.subheader("🌈 Imagery map")
                st.table(data["imagery_map"])
            if data.get("symbol_table"):
                st.subheader("🔶 Symbols")
                st.table(data["symbol_table"])

        if cat in ("play","story"):
            if show_line_by_line and cat == "play" and data.get("dialogue_beats"):
                st.subheader("💬 Dialogue beats")
                st.table(data["dialogue_beats"])
            if data.get("plot_points"):
                st.subheader("🧭 Plot points")
                st.table(data["plot_points"])
            if data.get("conflict"):
                st.subheader("⚔️ Conflict")
                st.write(data["conflict"])

        if cat == "essay":
            if data.get("thesis"):
                st.subheader("🎯 Thesis")
                st.write(data["thesis"])
            if data.get("key_points"):
                st.subheader("📌 Key points")
                st.table(data["key_points"])
            if data.get("rhetorical_devices"):
                st.subheader("✨ Rhetorical devices")
                st.table(data["rhetorical_devices"])

    # ----- Author & Characters -----
    with tabs[3]:
        if show_author and data.get("about_author"):
            st.subheader("✍️ About the Author")
            st.json(data["about_author"], expanded=False)
        if show_characters and data.get("characters"):
            st.subheader("👥 Character Sketches")
            rows = []
            for c in data["characters"]:
                if isinstance(c, dict):
                    rows.append({
                        "name": c.get("name",""),
                        "role": c.get("role",""),
                        "traits": ", ".join(c.get("traits",[])) if isinstance(c.get("traits"), list) else "",
                        "motives": ", ".join(c.get("motives",[])) if isinstance(c.get("motives"), list) else "",
                        "arc": c.get("arc_or_change","")
                    })
            st.table(rows or [{"name":"—","role":"—","traits":"—","motives":"—","arc":"—"}])

    # ----- Themes & Quotes -----
    with tabs[4]:
        if data.get("themes_detailed"):
            st.subheader("🧠 Themes & Evidence")
            rows = []
            for t in data["themes_detailed"]:
                if isinstance(t, dict):
                    ev = t.get("evidence_quotes",[])
                    ev_txt = " | ".join(ev) if isinstance(ev, list) else str(ev)
                    rows.append({"theme": t.get("theme",""), "explanation": t.get("explanation",""), "evidence": ev_txt})
            st.table(rows or [{"theme":"—","explanation":"—","evidence":"—"}])
        if show_activities and data.get("activities"):
            acts = data["activities"]
            for section in ["pre_reading","during_reading","post_reading","creative_tasks","projects"]:
                if acts.get(section):
                    st.subheader(f"🛠️ {section.replace('_',' ').title()}")
                    # editable
                    st.data_editor(acts[section], use_container_width=True, num_rows="dynamic")
        if show_rubric and data.get("assessment_rubric"):
            st.subheader("🧪 Assessment Rubric")
            st.json(data["assessment_rubric"], expanded=False)
        if data.get("vocabulary_glossary"):
            st.subheader("📒 Glossary")
            st.table(data["vocabulary_glossary"])
        if data.get("misconceptions"):
            st.subheader("⚠️ Misconceptions to avoid")
            st.write("\n".join(f"• {m}" for m in data["misconceptions"]))

    # ----- Q&A -----
    with tabs[5]:
        st.subheader("❔ Short Questions & Answers")
        srows = [{"q": s.get("q",""), "a": s.get("a","")} for s in qas.get("short",[])]
        st.table(srows or [{"q":"—","a":"—"}])

        st.subheader("🧩 Long Questions (Analytical) & Model Answers")
        lrows = [{"q": l.get("q",""), "a": l.get("a","")} for l in qas.get("long",[])]
        st.table(lrows or [{"q":"—","a":"—"}])

    # ----- Quiz -----
    with tabs[6]:
        st.subheader("🧠 Quiz (MCQs)")
        qrows = []
        for it in quiz:
            if isinstance(it, dict):
                qrows.append({
                    "q": it.get("q",""),
                    "choices": " | ".join(it.get("choices",[])) if isinstance(it.get("choices"), list) else str(it.get("choices","")),
                    "answer": it.get("answer","")
                })
            else:
                qrows.append({"q": str(it), "choices": "", "answer": ""})
        st.table(qrows or [{"q":"—","choices":"—","answer":"—"}])

    # ----- Export -----
    with tabs[7]:
        st.subheader("⬇️ Export")
        bundle = {"category": cat, "template": study_template, "data": data, "qas": qas, "quiz": quiz}

        # HTML
        html_str = build_portable_html(bundle)
        st.download_button(
            "Download portable HTML",
            data=html_str.encode("utf-8"),
            file_name=f"literature_{cat}_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

        # JSON
        st.markdown("#### Raw JSON (Template + Data + Q&As + Quiz)")
        st.json(bundle, expanded=False)
        st.download_button(
            "Download JSON (template + data + qas + quiz)",
            data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_{cat}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
