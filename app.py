# app.py
import os
import re
import json
from datetime import datetime

import requests
import streamlit as st

# --- Azure Doc Intelligence SDK (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight — Advanced", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Advanced)")
st.caption(
    "Upload/paste text → OCR → Auto-detect category → Category template → "
    "Detailed classroom-safe JSON (data + Q&As + quiz) → Export HTML/JSON."
)

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

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml` → AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT.")

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
# STRONG JSON PARSING (Hindi-safe)
# =========================
def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s

def extract_first_json_object(s: str) -> str:
    """
    Robustly extract the first top-level {...} JSON object from any text, even if
    the model printed extra prose around it.
    """
    if not s:
        return ""
    s = strip_code_fences(s)
    # Normalize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Find first '{' and walk to matching '}'
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return ""  # unmatched

def robust_parse_json(s: str):
    """
    Try direct parse → extract JSON block → strip trailing commas → parse.
    """
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    block = extract_first_json_object(s)
    if not block:
        return None
    # remove trailing commas before } or ]
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    try:
        return json.loads(block)
    except Exception:
        return None

# =========================
# OCR (IMAGES / PDFs)
# =========================
def ocr_read_any(bytes_blob: bytes) -> str:
    """Use Azure DI 'prebuilt-read' to extract text from images or PDFs."""
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
# CATEGORIES & TEMPLATES
# =========================
CATEGORIES = [
    "poetry", "play", "story", "essay", "biography", "autobiography",
    "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"
]

UNIVERSAL_SECTIONS = [
    "Title & Creator",
    "Introduction / Context",
    "Central Idea / Gist",
    "Summary (section-wise)",
    "Themes & Messages",
    "Tone & Mood",
    "Important Quotes / Key Lines",
    "Vocabulary & Meanings",
    "About Author/Poet/Speaker",
    "Question Bank (Short/Long/MCQ)",
    "Activities / Practice"
]

def make_study_template(category: str) -> dict:
    cat = category.lower()
    tpl = {"category": cat, "sections": list(UNIVERSAL_SECTIONS), "extras": []}

    if cat == "poetry":
        tpl["sections"].insert(5, "Structure (stanzas, rhyme, meter)")
        tpl["extras"] = ["Speaker/Voice", "Literary & Sound Devices", "Imagery & Symbolism", "Line-by-Line", "Emotional Arc"]
    elif cat == "play":
        tpl["sections"].insert(3, "Act/Scene-wise Summary")
        tpl["extras"] = ["Characters & Relationships", "Setting & Stage Directions", "Dialogue Beats", "Dramatic Devices", "Plot & Conflict"]
    elif cat == "story":
        tpl["extras"] = ["Narrative Voice", "Characters", "Setting", "Plot Points (exposition→resolution)", "Conflict"]
    elif cat == "essay":
        tpl["extras"] = ["Thesis/Main Claim", "Key Points with Evidence", "Structure", "Rhetorical Devices"]
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
        tpl["extras"] = ["Characters", "Setting", "Plot Outline", "Motifs/Symbols", "Moral / Cultural Significance"]

    return tpl

# =========================
# HEURISTIC CATEGORY GUESS
# =========================
def heuristic_guess_category(txt: str) -> str:
    t = txt.strip()
    lower = t.lower()
    if re.search(r'^\s*[A-Z].+\n[A-Z].+', t) and re.search(r'\n[A-Z][a-z]+:', t):
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
    return "story"

# =========================
# AZURE GPT CALL
# =========================
def call_azure_chat(messages, *, temperature=0.15, max_tokens=5500, force_json=True):
    """
    Calls Azure OpenAI Chat Completions with strict JSON (response_format=json_object).
    """
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code == 200:
            return True, r.json()["choices"][0]["message"]["content"]
        if r.status_code == 400 and "filtered" in r.text.lower():
            return False, "FILTERED"
        return False, f"Azure error {r.status_code}: {r.text[:600]}"
    except Exception as e:
        return False, f"Azure request failed: {e}"

# =========================
# CATEGORY SCHEMAS (DATA FILL)
# =========================
BASE_SAFE_FIELDS = {
    "language": "en|hi",
    "text_type": "category label",
    "literal_meaning": "plain-language meaning",
    "figurative_meaning": "themes/symbolism if any",
    "tone_mood": "tone & mood words",
    "one_sentence_takeaway": "classroom-safe summary",
    "executive_summary": "4–6 line student-friendly overview",
    "inspiration_hook": "engaging hook/analogy/activity",
    "why_it_matters": "how this connects to life/history/skills",
    "study_tips": ["bullet study tips"],
    "extension_reading": ["related readings/scenes/essays"],
    "emotional_arc": [{"beat": "setup|tension|turn|release", "feeling": "word", "evidence": "quote"}],
    "questions_short": [{"q": "1-2 line question", "a": "2-3 line answer"}],
    "questions_long": [{"q": "analytical question", "a": "5-8 line model answer with quotes"}]
}

ABOUT_AUTHOR = {
    "name": "if inferable",
    "era_or_period": "e.g., Elizabethan, Romantic, Modern, Bhakti, Progressive",
    "nationality_or_region": "if stated or inferable",
    "notable_works": ["..."],
    "themes_or_motifs": ["..."],
    "influences_or_style": "short note",
    "relevance_to_text": "how author context connects to text"
}

CHARACTER_SKETCH = [{
    "name": "...",
    "role": "protagonist/antagonist/support/narrator",
    "traits": ["..."],
    "motives": ["..."],
    "arc_or_change": "how the character evolves",
    "relationships": [{"with": "Name", "nature": "ally/rival/family", "note": "..."}],
    "key_quotes": [{"quote": "...", "explanation": "what it reveals"}]
}]

THEME_BLOCK = [{"theme": "...", "explanation": "...", "evidence_quotes": ["..."]}]

ACTIVITIES_BLOCK = {
    "pre_reading": [{"title": "...", "steps": ["...", "..."], "duration_min": 10}],
    "during_reading": [{"title": "...", "steps": ["..."], "strategy": "think-pair-share|annotation|jigsaw"}],
    "post_reading": [{"title": "...", "steps": ["..."], "outcome": "reflection|poster|debate"}],
    "creative_tasks": [{"title": "...", "type": "poem|skit|poster|diary|letter", "prompt": "..."}],
    "projects": [{"title": "...", "deliverable": "slide deck|poster|report|performance", "criteria": ["..."]}]
}

ASSESSMENT_RUBRIC = [{
    "criterion": "theme understanding|textual evidence|language analysis|presentation",
    "levels": {"exemplary": "descriptor", "proficient": "descriptor", "developing": "descriptor", "emerging": "descriptor"}
}]

# avoid walrus inside dict literals; bind once
ACTV = ACTIVITIES_BLOCK

SCHEMAS = {
    "poetry": {
        **BASE_SAFE_FIELDS,
        "speaker_or_voice": "who speaks / perspective",
        "structure_overview": {"stanzas": "count", "approx_line_count": "number", "rhyme_scheme": "e.g., ABAB", "meter_or_rhythm": "if notable"},
        "themes_detailed": THEME_BLOCK,
        "devices": [
            {"name": "Simile|Metaphor|Personification|Alliteration|Assonance|Consonance|Imagery|Symbolism|Hyperbole|Enjambment|Rhyme",
             "evidence": "quoted words", "explanation": "why it fits"}
        ],
        "imagery_map": [{"sense": "visual|auditory|tactile|gustatory|olfactory", "evidence": "quote", "effect": "reader impact"}],
        "symbol_table": [{"symbol": "...", "meaning": "...", "evidence": "..."}],
        "line_by_line": [{"line": "original line", "explanation": "meaning", "device_notes": "optional"}],
        "context_or_background": "poet/era/culture if relevant",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": ["..."],
        "quote_bank": ["..."],
        "comparative_texts": [{"title": "...", "note": "what to compare"}],
        "cross_curricular_links": [{"subject": "history|science|art", "idea": "..."}],
        "adaptation_ideas": ["stage reading|audio performance|visual collage"],
        "vocabulary_glossary": [{"term": "...", "meaning": "..."}],
        "misconceptions": ["..."]
    },
    "play": {
        **BASE_SAFE_FIELDS,
        "characters": CHARACTER_SKETCH,
        "setting": "time/place",
        "conflict": "internal/external and description",
        "dialogue_beats": [{"speaker": "Name", "line": "quoted dialogue", "note": "advance plot / reveal trait"}],
        "stage_directions": "if any (short)",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": ["..."],
        "quote_bank": ["..."],
        "comparative_texts": [{"title": "...", "note": "..."}],
        "cross_curricular_links": [{"subject": "...", "idea": "..."}],
        "adaptation_ideas": ["mini-drama|table read|radio play"]
    },
    "story": {
        **BASE_SAFE_FIELDS,
        "narrative_voice": "first/third/omniscient/limited",
        "setting": "time/place",
        "characters": CHARACTER_SKETCH,
        "plot_points": [{"stage": "exposition|rising|climax|falling|resolution", "what_happens": "...", "evidence": "quote/line"}],
        "conflict": "type + description",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": ["..."],
        "quote_bank": ["..."],
        "comparative_texts": [{"title": "...", "note": "..."}],
        "cross_curricular_links": [{"subject": "...", "idea": "..."}],
        "adaptation_ideas": ["storyboard|podcast|comic strip"]
    },
    "essay": {
        **BASE_SAFE_FIELDS,
        "thesis": "author's main claim",
        "key_points": [{"point": "...", "evidence_or_example": "...", "counterpoint_if_any": "optional"}],
        "structure": "intro/body/conclusion notes",
        "tone_register": "formal/informal/analytical",
        "rhetorical_devices": [{"name": "analogy|contrast|examples|statistics|allusion", "evidence": "...", "effect": "..."}],
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC,
        "homework": ["..."],
        "quote_bank": ["..."]
    },
    "biography": {
        **BASE_SAFE_FIELDS,
        "subject": "person",
        "timeline": [{"year_or_age": "...", "event": "...", "impact": "..."}],
        "qualities": ["..."],
        "influence_or_impact": "...",
        "notable_works_or_contributions": ["..."],
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "autobiography": {
        **BASE_SAFE_FIELDS,
        "author": "person",
        "episodes": [{"when": "...", "event": "...", "reflection": "...", "lesson": "..."}],
        "themes_detailed": THEME_BLOCK,
        "voice_and_style": "...",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "speech": {
        **BASE_SAFE_FIELDS,
        "audience": "who",
        "purpose": "inform/persuade/inspire",
        "key_points": ["..."],
        "rhetorical_devices": [{"name": "repetition|anaphora|rhetorical_question|parallelism|allusion", "evidence": "...", "effect": "..."}],
        "call_to_action": "if any",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "letter": {
        **BASE_SAFE_FIELDS,
        "letter_type": "formal|informal",
        "salutation": "...",
        "body_points": [{"point": "...", "example_or_reason": "..."}],
        "closing": "...",
        "tone_register": "polite/warm/requesting/complaint",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "diary": {
        **BASE_SAFE_FIELDS,
        "date_or_time_hint": "if present",
        "events": ["..."],
        "feelings": "emotion words",
        "reflection": "what was learned",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "report": {
        **BASE_SAFE_FIELDS,
        "topic": "...",
        "sections": [{"heading": "Introduction|Method|Observation|Discussion|Conclusion", "summary": "..."}],
        "findings": ["..."],
        "recommendations": ["..."],
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "folk_tale": {
        **BASE_SAFE_FIELDS,
        "characters": CHARACTER_SKETCH,
        "setting": "...",
        "plot_outline": ["..."],
        "repeating_patterns_or_motifs": ["..."],
        "moral_or_lesson": "...",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "myth": {
        **BASE_SAFE_FIELDS,
        "deities_or_symbols": ["..."],
        "origin_or_explanation": "what it explains",
        "plot_outline": ["..."],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "legend": {
        **BASE_SAFE_FIELDS,
        "hero_or_figure": "...",
        "historical_backdrop": "...",
        "notable_events": ["..."],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTV,
        "assessment_rubric": ASSESSMENT_RUBRIC
    }
}

# =========================
# PROMPTS
# =========================
def classify_with_gpt(txt: str, lang: str) -> str:
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

def build_schema_prompt(category: str, language_code: str, detail: int, n_short: int, n_long: int, n_quiz: int) -> str:
    """
    Strictly require a single top-level JSON object:
    {
      "data": {...},
      "qas": {"short": [...], "long": [...]},
      "quiz": [{"q":"...", "choices":["A","B","C","D"], "answer":"A"}]
    }
    The "data" object uses SCHEMAS[category].
    """
    schema = dict(SCHEMAS.get(category, SCHEMAS["story"]))  # shallow copy

    hard_rules = f"""
STRICT OUTPUT (OBEY):
- Return ONLY one JSON object. No markdown, no prose.
- Top-level keys: "data", "qas", "quiz".
- "data" MUST follow the provided schema keys (omit keys that don't apply).
- "qas.short" length: {n_short}. "qas.long" length: {n_long}. "quiz" length: {n_quiz}.
- For poetry or lyrics: ALWAYS provide "line_by_line" with actual lines (quote exact line fragments) and explanations.
- If target language is hi, explanations may be in Hindi; quotes MUST stay true to the text; if text is English keep quotes in English.
- Keep answers classroom-safe; cite short quotes where helpful.
- Be concise but complete (detail level {detail}/5).
"""
    return (
        "You are an expert literature teacher. Fill DATA + QAs + quiz for the given text.\n"
        "Return ONLY JSON with this exact shape: {\"data\": {...}, \"qas\": {\"short\": [...], \"long\": [...]}, \"quiz\": [...]}.\n"
        f"Target language: {language_code}\nCategory: {category}\n"
        + hard_rules +
        "\nSCHEMA for the \"data\" object (guide keys):\n" + json.dumps(schema, ensure_ascii=False, indent=2)
    )

# =========================
# UI INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a poem/play/story/essay (optional)", height=160, placeholder="e.g., Two roads diverged in a yellow wood…")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols_top = st.columns(6)
with cols_top[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols_top[1]:
    detail_level = st.slider("Detail level", 1, 5, 5, help="Higher = richer explanations (still concise).")
with cols_top[2]:
    short_q_count = st.slider("Short Q&A (count)", 1, 10, 4)
with cols_top[3]:
    long_q_count = st.slider("Long Q&A (count)", 1, 10, 4)
with cols_top[4]:
    quiz_count = st.slider("Quiz MCQs (count)", 4, 20, 10, step=2)
with cols_top[5]:
    allow_padding = st.toggle("Pad missing lists", value=False, help="If on, app pads some lists with generic items to reach counts.")

# Display toggles
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
    show_quotes = st.toggle("Quote Bank", value=True)

run = st.button("🔎 Analyze (Advanced)")

# =========================
# OPTIONAL ENFORCER (simple)
# =========================
def _pad_list(lst, n, filler):
    if not isinstance(lst, list):
        lst = []
    out = list(lst)
    while len(out) < n:
        out.append(filler(out[-1] if out else None, len(out)))
    return out

def enforce_minimums(data: dict, n: int) -> dict:
    if not isinstance(data, dict):
        return {}
    # themes_detailed
    if "themes_detailed" in data:
        def filler_theme(prev, idx):
            return {"theme": f"Theme {idx+1}", "explanation": "Reinforced insight.", "evidence_quotes": []}
        arr = data.get("themes_detailed")
        data["themes_detailed"] = _pad_list(arr if isinstance(arr, list) else [], n, filler_theme)
    # quote_bank
    if "quote_bank" in data:
        def filler_quote(prev, idx):
            return "Additional supporting line (reinforced)"
        arr = data.get("quote_bank")
        data["quote_bank"] = _pad_list(arr if isinstance(arr, list) else [], n, filler_quote)
    # emotional_arc
    if "emotional_arc" in data:
        def filler_arc(prev, idx):
            beats = ["setup","tension","turn","release","afterglow"]
            return {"beat": beats[idx % len(beats)], "feeling": "reflective", "evidence": "(reinforced)"}
        arr = data.get("emotional_arc")
        data["emotional_arc"] = _pad_list(arr if isinstance(arr, list) else [], n, filler_arc)
    return data

# =========================
# HTML EXPORT
# =========================
def build_portable_html(bundle: dict) -> str:
    cat = bundle.get("category","—")
    data = bundle.get("data",{})
    qas = bundle.get("qas",{})
    quiz = bundle.get("quiz",[])
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    def esc(x): return (x or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    def list_html(items):
        if not items: return "<p>—</p>"
        lis = "".join(f"<li>{esc(str(i))}</li>" for i in items)
        return f"<ul>{lis}</ul>"
    theme_rows = ""
    for t in data.get("themes_detailed",[]) or []:
        ev = t.get("evidence_quotes",[])
        theme_rows += f"<tr><td>{esc(t.get('theme',''))}</td><td>{esc(t.get('explanation',''))}</td><td>{esc(' | '.join(ev or []))}</td></tr>"
    html = f"""<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Literature Insight — {esc(cat.title())}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;line-height:1.6;margin:24px;color:#0f172a}}
h1,h2,h3{{margin:0.4em 0}} .card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;background:#fff}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}}
.small{{color:#475569;font-size:12px}}
</style></head><body>
<h1>Literature Insight — {esc(cat.title())}</h1>
<div class="small">Generated: {ts}</div>

<div class="card"><h2>Executive Summary</h2><p>{esc(data.get('executive_summary') or data.get('one_sentence_takeaway') or '—')}</p></div>
<div class="card"><h2>Inspiration Hook</h2><p>{esc(data.get('inspiration_hook') or '—')}</p></div>
<div class="card"><h2>Why It Matters</h2><p>{esc(data.get('why_it_matters') or '—')}</p></div>
<div class="card"><h2>Study Tips</h2>{list_html(data.get('study_tips') or [])}</div>

<div class="card"><h2>Themes & Evidence</h2>
<table><thead><tr><th>Theme</th><th>Explanation</th><th>Evidence</th></tr></thead>
<tbody>{theme_rows or '<tr><td colspan="3">—</td></tr>'}</tbody></table></div>

<div class="card"><h2>Quote Bank</h2>{list_html(data.get('quote_bank') or [])}</div>

<div class="card"><h2>Short Q&A</h2>{list_html([f"Q: {qa.get('q','')} — A: {qa.get('a','')}" for qa in (qas.get('short') or [])])}</div>
<div class="card"><h2>Long Q&A</h2>{list_html([f"Q: {qa.get('q','')} — A: {qa.get('a','')}" for qa in (qas.get('long') or [])])}</div>
<div class="card"><h2>Quiz</h2>{list_html([f"{i+1}. {q.get('q','')} (Ans: {q.get('answer','?')})" for i,q in enumerate(quiz)])}</div>

<div class="small">© Suvichaar Literature Insight</div>
</body></html>"""
    return html

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

    st.markdown("#### Detected")
    st.markdown(f"<span style='padding:2px 10px;border-radius:999px;background:#0891b2;color:#fff;'>Language: {explain_lang}</span>", unsafe_allow_html=True)
    st.write("")

    # Category detection (heuristic + GPT)
    guessed = heuristic_guess_category(source_text)
    try:
        cat = classify_with_gpt(source_text, explain_lang) or guessed
    except Exception:
        cat = guessed
    st.markdown(f"<span style='padding:2px 10px;border-radius:999px;background:#16a34a;color:#fff;'>Category: {cat}</span>", unsafe_allow_html=True)

    # Category study template
    study_template = make_study_template(cat)

    # Prompt & call
    safe_text = make_classroom_safe(source_text)
    system_msg = (
        "You are a veteran literature teacher. "
        "You must return STRICT JSON only. No extra text."
        + (" Respond in Hindi for explanations (quotes keep original language)." if explain_lang.startswith("hi") else " Respond in English.")
    )
    user_msg = (
        f"TEXT TO ANALYZE (verbatim):\n{safe_text}\n\n" +
        build_schema_prompt(cat, explain_lang, detail_level, short_q_count, long_q_count, quiz_count)
    )

    with st.spinner("Calling Azure for detailed data + Q&As + quiz…"):
        ok, content = call_azure_chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2 if detail_level >= 4 else 0.1,
            max_tokens=6000,
            force_json=True
        )

    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in tighter student-safe mode…")
        ok, content = call_azure_chat(
            [{"role": "system", "content": "Return ONLY JSON. Avoid explicit terms; use neutral wording."},
             {"role": "user", "content": f"Return ONLY JSON with keys data/qas/quiz for category '{cat}', language {explain_lang}. Text:\n{safe_text}"}],
            temperature=0.0, max_tokens=5000, force_json=True
        )

    if not ok:
        st.error(content)
        st.stop()

    parsed = robust_parse_json(content)
    if not isinstance(parsed, dict):
        st.error("Model did not return valid JSON.")
        st.stop()

    data = parsed.get("data") or {}
    qas = parsed.get("qas") or {"short": [], "long": []}
    quiz = parsed.get("quiz") or []

    # Optional padding
    if allow_padding:
        data = enforce_minimums(data, 3)
        # pad qas/quiz lengths if needed
        def fill_short(prev, idx):
            return {"q": f"Additional short question {idx}", "a": "A concise answer (reinforced)."}
        def fill_long(prev, idx):
            return {"q": f"Additional analytical question {idx}", "a": "A thoughtful, evidence-based answer (reinforced)."}
        def pad_list(lst, n, filler):
            return _pad_list(lst if isinstance(lst, list) else [], n, filler)
        qas["short"] = pad_list(qas.get("short"), short_q_count, fill_short)
        qas["long"]  = pad_list(qas.get("long"),  long_q_count,  fill_long)
        def fill_mcq(prev, idx):
            return {"q": f"Extra MCQ {idx}", "choices": ["A","B","C","D"], "answer":"A"}
        quiz = pad_list(quiz, quiz_count, fill_mcq)

    # ============= PRESENTATION LAYOUT =============
    st.markdown("---")
    tabs = st.tabs([
        "Study Template", "Overview", "Text Insights", "Author & Characters",
        "Themes & Quotes", "Activities & Rubrics", "Q&A + Emotional Arc",
        "Quiz", "Export / JSON"
    ])

    # ----- Study Template -----
    with tabs[0]:
        st.markdown("### 🧩 Category-specific Study Template")
        st.write(f"**Category:** {study_template['category'].title()}")
        st.markdown("**Core Sections**")
        st.write("\n".join(f"• {s}" for s in study_template["sections"]))
        if study_template.get("extras"):
            st.markdown("**Extras (focus for this category)**")
            st.write("\n".join(f"• {s}" for s in study_template["extras"]))

    # ----- Overview -----
    with tabs[1]:
        st.markdown("### 🧭 Executive Summary")
        st.write(data.get("executive_summary") or data.get("one_sentence_takeaway") or "—")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✨ Inspiration Hook")
            st.write(data.get("inspiration_hook") or "—")
        with c2:
            st.markdown("### 🎯 Why It Matters")
            st.write(data.get("why_it_matters") or "—")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 📝 Study Tips")
            tips = data.get("study_tips")
            st.write("\n".join(f"• {x}" for x in (tips or [])) or "—")
        with c4:
            st.markdown("### 📚 Extension Reading")
            ext = data.get("extension_reading")
            st.write("\n".join(f"• {x}" for x in (ext or [])) or "—")

    # ----- Text Insights -----
    with tabs[2]:
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("### 📘 Literal Meaning")
            st.write(data.get("literal_meaning","—"))
            st.markdown("### 🌊 Figurative Meaning / Themes")
            st.write(data.get("figurative_meaning","—"))
        with rcol:
            st.markdown("### 🎼 Tone & Mood")
            st.write(data.get("tone_mood","—"))
            st.markdown("### ✅ One-sentence Takeaway")
            st.write(data.get("one_sentence_takeaway","—"))

        if cat == "poetry":
            if data.get("structure_overview"):
                st.markdown("### 🧩 Structure Overview")
                st.json(data["structure_overview"], expanded=False)
            if show_line_by_line:
                lbl = data.get("line_by_line")
                st.markdown("### 📖 Line-by-Line / Section-wise")
                if isinstance(lbl, list) and lbl:
                    st.table([{
                        "line": it.get("line",""),
                        "explanation": it.get("explanation",""),
                        "device_notes": it.get("device_notes","")
                    } for it in lbl])
                else:
                    st.info("No line-by-line items were extracted from the text.")
            if show_devices_table and data.get("devices"):
                st.markdown("### 🎭 Devices")
                devs = data.get("devices") or []
                # Guard for non-dict entries
                rows = []
                for d in devs:
                    if isinstance(d, dict):
                        rows.append({"device": d.get("name",""), "evidence": d.get("evidence",""), "why": d.get("explanation","")})
                    else:
                        rows.append({"device": str(d), "evidence": "", "why": ""})
                st.table(rows)
            if data.get("imagery_map"):
                st.markdown("### 🌈 Imagery Map")
                st.table(data["imagery_map"])
            if data.get("symbol_table"):
                st.markdown("### 🔶 Symbols")
                st.table(data["symbol_table"])

        if cat in ("play","story"):
            if show_line_by_line and cat == "play" and data.get("dialogue_beats"):
                st.markdown("### 💬 Dialogue Beats")
                st.table(data["dialogue_beats"])
            if data.get("plot_points"):
                st.markdown("### 🧭 Plot Points")
                st.table(data["plot_points"])
            if data.get("conflict"):
                st.markdown("### ⚔️ Conflict")
                st.write(data["conflict"])

        if cat == "essay":
            if data.get("thesis"):
                st.markdown("### 🎯 Thesis")
                st.write(data["thesis"])
            if data.get("key_points"):
                st.markdown("### 📌 Key Points")
                st.table(data["key_points"])
            if data.get("rhetorical_devices"):
                st.markdown("### ✨ Rhetorical Devices")
                st.table(data["rhetorical_devices"])

    # ----- Author & Characters -----
    with tabs[3]:
        if show_author and data.get("about_author"):
            st.markdown("### ✍️ About the Author")
            st.json(data["about_author"], expanded=False)

        if show_characters and data.get("characters"):
            st.markdown("### 👥 Character Sketches")
            chars = data.get("characters") or []
            st.table([{
                "name": c.get("name",""),
                "role": c.get("role",""),
                "traits": ", ".join(c.get("traits",[])) if isinstance(c.get("traits"), list) else c.get("traits",""),
                "motives": ", ".join(c.get("motives",[])) if isinstance(c.get("motives"), list) else c.get("motives",""),
                "arc": c.get("arc_or_change","")
            } for c in chars])
            for c in chars:
                if isinstance(c, dict) and c.get("key_quotes"):
                    st.markdown(f"**Key quotes — {c.get('name','Character')}**")
                    st.write("\n".join(f"• {q.get('quote','')} — {q.get('explanation','')}" for q in c["key_quotes"]))

    # ----- Themes & Quotes -----
    with tabs[4]:
        if data.get("themes_detailed"):
            st.markdown("### 🧠 Themes & Evidence")
            rows = []
            for t in data.get("themes_detailed",[]):
                if isinstance(t, dict):
                    ev = t.get("evidence_quotes", [])
                    rows.append({
                        "theme": t.get("theme",""),
                        "explanation": t.get("explanation",""),
                        "evidence": " | ".join(ev) if isinstance(ev, list) else ev
                    })
            st.table(rows)
        if show_quotes and data.get("quote_bank"):
            st.markdown("### ❝ Quote Bank")
            st.write("\n".join(f"• {q}" for q in data["quote_bank"]))

        if data.get("comparative_texts"):
            st.markdown("### 🔁 Comparative Texts")
            st.table(data["comparative_texts"])
        if data.get("cross_curricular_links"):
            st.markdown("### 🔗 Cross-curricular Links")
            st.table(data["cross_curricular_links"])
        if data.get("adaptation_ideas"):
            st.markdown("### 🎬 Adaptation Ideas")
            st.write("\n".join(f"• {a}" for a in data["adaptation_ideas"]))

    # ----- Activities & Rubrics -----
    with tabs[5]:
        if show_activities and data.get("activities"):
            acts = data["activities"]
            for section in ["pre_reading","during_reading","post_reading","creative_tasks","projects"]:
                if acts.get(section):
                    st.markdown(f"### 🛠️ {section.replace('_',' ').title()}")
                    for item in acts[section]:
                        st.markdown(f"- **{item.get('title','Activity')}**")
                        if "steps" in item and isinstance(item["steps"], list):
                            for i, sstep in enumerate(item["steps"], start=1):
                                st.write(f"  {i}. {sstep}")
                        for k in ("duration_min","strategy","type","deliverable","criteria","outcome","prompt"):
                            if item.get(k):
                                st.caption(f"{k}: {item[k]}")
        if show_rubric and data.get("assessment_rubric"):
            st.markdown("### 🧪 Assessment Rubric")
            st.json(data["assessment_rubric"], expanded=False)

        if data.get("vocabulary_glossary"):
            st.markdown("### 📒 Glossary")
            st.table(data["vocabulary_glossary"])
        if data.get("misconceptions"):
            st.markdown("### ⚠️ Misconceptions to avoid")
            st.write("\n".join(f"• {m}" for m in data["misconceptions"]))

    # ----- Q&A + Emotional Arc -----
    with tabs[6]:
        if data.get("emotional_arc"):
            st.markdown("### 💓 Emotional Arc (Build & Release)")
            st.table(data["emotional_arc"])

        if qas.get("short"):
            st.markdown("### ❔ Short Questions & Answers")
            st.table(qas["short"])

        if qas.get("long"):
            st.markdown("### 🧩 Long Questions (Analytical) & Model Answers")
            st.table(qas["long"])

    # ----- Quiz -----
    with tabs[7]:
        st.markdown("### 🧠 Quiz (MCQs)")
        if quiz:
            st.table([{
                "q": q.get("q",""),
                "choices": " | ".join(q.get("choices",[])),
                "answer": q.get("answer","")
            } for q in quiz])
        else:
            st.write("—")

    # ----- Export / JSON -----
    with tabs[8]:
        bundle = {"category": cat, "template": study_template, "data": data, "qas": qas, "quiz": quiz}

        st.markdown("### ⬇️ Download HTML snapshot")
        html_str = build_portable_html(bundle)
        st.download_button(
            "Download portable HTML",
            data=html_str.encode("utf-8"),
            file_name=f"literature_{cat}_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        st.markdown("### 🧾 Raw JSON (Template + Data + Q&As + Quiz)")
        st.json(bundle, expanded=False)
        st.download_button(
            "Download JSON (template + data + qas + quiz)",
            data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_{cat}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
