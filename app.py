# app.py
# Suvichaar Literature Insight — Option B (Optimized)
# ---------------------------------------------------
# - Category detection (heuristic + Azure fallback)
# - Category-specific study template (deterministic)
# - Structured data generation via Azure OpenAI (JSON-only)
# - Robust JSON repair + safe fallbacks
# - Optional padding to reach evidence counts (off by default)
# - Clean UI with clear sections

import re
import json
from datetime import datetime

import requests
import streamlit as st

# --- Azure Doc Intelligence SDK (OCR) (optional) ---
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
    "Paste/upload text → Auto-detect category → Get a category-specific study template + classroom-safe, structured data "
    "(summary, themes, devices, characters, emotional arc, Q&A, activities, rubric)."
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
# UTIL: CLASSROOM-SAFE SANITIZER
# =========================
def make_classroom_safe(text: str) -> str:
    """Replace potentially sensitive words with gentler terms to reduce filtering issues."""
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
# AZURE OPENAI CALLERS
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=4800, force_json=False):
    """Generic Azure Chat Completions call."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        # Uses the preview response_format that asks model to emit strict JSON
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


def robust_json_parse(s: str):
    """Try normal parse → repair → None."""
    if not s:
        return None
    # 1) try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) repair
    s2 = s.strip()
    # strip code fences
    s2 = re.sub(r"^```(?:json)?\s*", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```$", "", s2)
    # smart quotes
    s2 = s2.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # trailing commas
    s2 = re.sub(r",(\s*[}\]])", r"\1", s2)
    # extract first JSON object
    m = re.search(r"\{[\s\S]*\}", s2)
    if m:
        s2 = m.group(0)
    try:
        return json.loads(s2)
    except Exception:
        return None


# =========================
# OCR (IMAGE/PDF) VIA AZURE DI (OPTIONAL)
# =========================
def ocr_read_any(bytes_blob: bytes) -> str:
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
                if lines:
                    parts.append("\n".join(lines).strip())
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
# LANGUAGE & CATEGORY
# =========================
def detect_hi_or_en(text: str) -> str:
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total = devanagari + latin
    if total == 0:
        return "en"
    return "hi" if (devanagari / total) >= 0.25 else "en"


CATEGORIES = [
    "poetry", "play", "story", "essay", "biography", "autobiography",
    "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"
]

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


def classify_with_gpt(txt: str) -> str:
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
        temperature=0.0, max_tokens=16, force_json=False
    )
    if not ok or not out:
        return heuristic_guess_category(txt)
    label = re.sub(r'[^a-z_]', '', out.strip().lower())
    return label if label in CATEGORIES else heuristic_guess_category(txt)


# =========================
# STUDY TEMPLATE (DETERMINISTIC)
# =========================
UNIVERSAL_SECTIONS = [
    "Title & Creator",
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
    cat = category.lower()
    tpl = {"category": cat, "sections": list(UNIVERSAL_SECTIONS), "extras": []}

    if cat == "poetry":
        tpl["sections"].insert(5, "Structure (stanzas, rhyme scheme, meter)")
        tpl["extras"] = ["Speaker/Voice", "Literary & Sound Devices", "Imagery & Symbolism", "Emotional Arc"]
    elif cat == "play":
        tpl["sections"].insert(3, "Scene/Act-wise Summary")
        tpl["extras"] = ["Characters & Relationships", "Setting & Stage Directions", "Dialogue Beats", "Dramatic Devices", "Plot Points & Conflict"]
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
# CATEGORY SCHEMAS (for DATA generation)
# =========================
BASE_SAFE_FIELDS = {
    "language": "en|hi",
    "text_type": "category label",
    "literal_meaning": "plain-language meaning",
    "figurative_meaning": "themes/symbolism if any",
    "tone_mood": "tone & mood words",
    "one_sentence_takeaway": "classroom-safe summary",
    "executive_summary": "4–6 line high-level overview for students",
    "inspiration_hook": "an engaging hook/activity/analogy to spark curiosity",
    "why_it_matters": "how this text connects to life/history/skills",
    "study_tips": ["bullet study tips"],
    "extension_reading": ["related readings, scenes, essays"],
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
TEACHER_VIEW = {
    "learning_objectives": ["..."],
    "discussion_questions": ["..."],
    "quick_assessment_mcq": [{"q": "...", "choices": ["A", "B", "C", "D"], "answer": "A"}]
}

SCHEMAS = {
    "poetry": {
        **BASE_SAFE_FIELDS,
        "speaker_or_voice": "who speaks / perspective",
        "structure_overview": {"stanzas": "count", "approx_line_count": "number", "rhyme_scheme": "e.g., ABAB", "meter_or_rhythm": "if notable"},
        "themes_detailed": THEME_BLOCK,
        "devices": [{"name": "Simile|Metaphor|Personification|Alliteration|Assonance|Consonance|Imagery|Symbolism|Hyperbole|Enjambment|Rhyme", "evidence": "quoted words", "explanation": "why it fits"}],
        "imagery_map": [{"sense": "visual|auditory|tactile|gustatory|olfactory", "evidence": "quote", "effect": "reader impact"}],
        "symbol_table": [{"symbol": "...", "meaning": "...", "evidence": "..."}],
        "line_by_line": [{"line": "original line", "explanation": "meaning", "device_notes": "optional"}],
        "context_or_background": "poet/era/culture if relevant",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTIVITIES_BLOCK,
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
        "dialogue_beats": [{"speaker": "Name", "line": "quoted dialogue", "note": "function"}],
        "stage_directions": "if any (short)",
        "themes_detailed": THEME_BLOCK,
        "about_author": ABOUT_AUTHOR,
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "autobiography": {
        **BASE_SAFE_FIELDS,
        "author": "person",
        "episodes": [{"when": "...", "event": "...", "reflection": "...", "lesson": "..."}],
        "themes_detailed": THEME_BLOCK,
        "voice_and_style": "...",
        "about_author": ABOUT_AUTHOR,
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "diary": {
        **BASE_SAFE_FIELDS,
        "date_or_time_hint": "if present",
        "events": ["..."],
        "feelings": "emotion words",
        "reflection": "what was learned",
        "themes_detailed": THEME_BLOCK,
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "report": {
        **BASE_SAFE_FIELDS,
        "topic": "...",
        "sections": [{"heading": "Introduction|Method|Observation|Discussion|Conclusion", "summary": "..."}],
        "findings": ["..."],
        "recommendations": ["..."],
        "activities": ACTIVITIES_BLOCK,
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
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "myth": {
        **BASE_SAFE_FIELDS,
        "deities_or_symbols": ["..."],
        "origin_or_explanation": "what it explains",
        "plot_outline": ["..."],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    },
    "legend": {
        **BASE_SAFE_FIELDS,
        "hero_or_figure": "...",
        "historical_backdrop": "...",
        "notable_events": ["..."],
        "themes_detailed": THEME_BLOCK,
        "activities": ACTIVITIES_BLOCK,
        "assessment_rubric": ASSESSMENT_RUBRIC
    }
}


def build_schema_prompt(category: str, language_code: str, detail: int, evidence_count: int, teacher_mode: bool) -> str:
    """Prompt: ask for ONLY { "data": { ... } } following the schema."""
    schema = dict(SCHEMAS.get(category, SCHEMAS["story"]))  # shallow copy is fine
    if teacher_mode:
        schema["teacher_view"] = {
            "learning_objectives": ["..."],
            "discussion_questions": ["..."],
            "quick_assessment_mcq": [{"q": "...", "choices": ["A", "B", "C", "D"], "answer": "A"}]
        }

    hard_rules = f"""
STRICT OUTPUT REQUIREMENTS:
- Return ONLY a JSON object with a single top-level key: "data".
- "data" must use the provided schema keys. Omit keys that do not apply to the text.
- For each list/array, provide AT MOST {evidence_count} items. Keep entries concise.
- Do NOT invent placeholder items. If the text supports fewer, return fewer.
- Classroom-safe wording; quote evidence verbatim where asked.

Emphasize detailed, specific content for:
- "executive_summary": 6–9 sentences covering setting, characters, conflict/tension, and foreshadowing.
- "inspiration_hook": 2–3 vivid lines, classroom-friendly.
- "why_it_matters": 3–4 lines linking to craft (e.g., tension building), life skills, and reader empathy.
- "study_tips": 5–7 concrete, actionable bullets.
- "extension_reading": 4–6 curated items (mix: primary text, criticism, historical context).
"""

    return (
        "You are an expert literature teacher. Fill study DATA for the given text.\n"
        "Return ONLY JSON with this exact shape: {\"data\": { ... }} (no markdown, no comments).\n"
        f"Target language: {language_code}\nCategory: {category}\n"
        f"Detail level: {detail} (richer but concise). Item budget per list: ≤ {evidence_count}\n"
        + hard_rules +
        "\nSCHEMA for the 'data' object:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
    )


# =========================
# OPTIONAL: MINIMUM ENFORCER (only when user enables toggle)
# =========================
def _pad_list(lst, n, filler):
    if not isinstance(lst, list):
        lst = []
    out = list(lst)
    while len(out) < n:
        out.append(filler(out[-1] if out else None, len(out)))
    return out

def _split_or_clone_quotes(item, target_len):
    """Split multi-sentence evidence into multiple items before cloning."""
    if not item:
        return []
    text = ""
    if isinstance(item, str):
        text = item
    elif isinstance(item, dict):
        for k in ("evidence", "quote", "evidence_quotes"):
            v = item.get(k)
            if isinstance(v, str):
                text = v
                break
            if isinstance(v, list) and v:
                text = " ".join(v)
                break
    if not text or target_len <= 1:
        return [item]
    chunks = re.split(r"\s*(?:\|\||\|\.|[.;]\s+|\n)\s*", text)
    chunks = [c.strip(" |") for c in chunks if c.strip()]
    if len(chunks) >= target_len:
        if isinstance(item, dict):
            items = []
            for c in chunks[:target_len]:
                new = dict(item)
                if "evidence_quotes" in new and isinstance(new["evidence_quotes"], list):
                    new["evidence_quotes"] = [c]
                elif "evidence" in new:
                    new["evidence"] = c
                elif "quote" in new:
                    new["quote"] = c
                items.append(new)
            return items
        else:
            return chunks[:target_len]
    return [item]

def enforce_minimums(data: dict, n: int) -> dict:
    """Pad a few lists to n items (only if toggle is ON)."""
    if not isinstance(data, dict):
        return {}

    def filler_quote(prev, idx):
        if isinstance(prev, str):
            return prev + " (reinforced)"
        return (prev or "—")

    def filler_theme(prev, idx):
        base = {"theme": f"Theme {idx+1}", "explanation": "Reinforced insight.", "evidence_quotes": []}
        if isinstance(prev, dict):
            p = dict(prev)
            p["explanation"] = (p.get("explanation", "") + " (reinforced)").strip()
            return p
        return base

    # themes_detailed
    themes = data.get("themes_detailed")
    if isinstance(themes, list) and themes:
        expanded = []
        for t in themes:
            items = _split_or_clone_quotes(t, n)
            expanded.extend(items)
        if len(expanded) < n:
            expanded = _pad_list(expanded, n, filler_theme)
        data["themes_detailed"] = expanded[:max(n, len(expanded))]
    elif themes is not None:
        data["themes_detailed"] = _pad_list([], n, filler_theme)

    # quote_bank
    if "quote_bank" in data:
        qb = data.get("quote_bank")
        if isinstance(qb, list) and qb:
            expanded = []
            for q in qb:
                expanded.extend(_split_or_clone_quotes(q, n))
            if len(expanded) < n:
                expanded = _pad_list(expanded, n, filler_quote)
            data["quote_bank"] = expanded[:max(n, len(expanded))]
        else:
            data["quote_bank"] = _pad_list([], n, filler_quote)

    # emotional_arc
    if "emotional_arc" in data:
        ea = data.get("emotional_arc")
        def filler_arc(prev, idx):
            beats = ["setup", "tension", "turn", "release", "afterglow"]
            return {"beat": beats[idx % len(beats)], "feeling": "reflective", "evidence": "(reinforced)"}
        data["emotional_arc"] = _pad_list(ea if isinstance(ea, list) else [], n, filler_arc)

    # QAs
    for key, short in (("questions_short", True), ("questions_long", False)):
        def filler_qa(prev, idx, long=not short):
            return {
                "q": f"Additional {'analytical ' if long else ''}question {idx+1}",
                "a": ("A thoughtful, evidence-based answer (reinforced)." if long
                      else "A concise answer with a brief quote (reinforced).")
            }
        arr = data.get(key)
        data[key] = _pad_list(arr if isinstance(arr, list) else [], n, lambda p, i: filler_qa(p, i))

    # activities
    if "activities" in data and isinstance(data["activities"], dict):
        for sect in ["pre_reading", "during_reading", "post_reading"]:
            def filler_act(prev, idx, tag=sect):
                base = {"title": f"Extra {tag.replace('_', ' ').title()} {idx+1}", "steps": ["Add one more step."], "duration_min": 8}
                if tag == "during_reading":
                    base["strategy"] = "annotation"
                if tag == "post_reading":
                    base["outcome"] = "reflection"
                return base
            arr = data["activities"].get(sect)
            data["activities"][sect] = _pad_list(arr if isinstance(arr, list) else [], n, lambda p, i: filler_act(p, i))

    # teacher_view
    if "teacher_view" in data and isinstance(data["teacher_view"], dict):
        def pad_simple_list(name):
            arr = data["teacher_view"].get(name)
            def filler(prev, idx):
                return f"Additional {name.replace('_', ' ')} {idx+1} (reinforced)"
            data["teacher_view"][name] = _pad_list(arr if isinstance(arr, list) else [], n, filler)

        pad_simple_list("discussion_questions")

        mcq = data["teacher_view"].get("quick_assessment_mcq")
        def filler_mcq(prev, idx):
            return {"q": f"Extra MCQ {idx+1}", "choices": ["A", "B", "C", "D"], "answer": "A"}
        data["teacher_view"]["quick_assessment_mcq"] = _pad_list(mcq if isinstance(mcq, list) else [], n, lambda p, i: filler_mcq(p, i))

    return data


# =========================
# UI HELPERS
# =========================
def chip(text: str, color: str = "#2563eb"):
    st.markdown(
        f"""<span style="
            display:inline-block;
            padding:2px 10px;
            margin-right:6px;
            border-radius:999px;
            background:{color};color:white;font-size:12px;">{text}</span>""",
        unsafe_allow_html=True
    )

def h2(title: str, emoji: str = "🔹"):
    st.markdown(f"### {emoji} {title}")

def safe_join_bullets(items):
    if not items:
        return "—"
    return "\n".join(f"• {x}" for x in items if isinstance(x, str) and x.strip())

def build_portable_html(data: dict, category: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    def esc(x):
        return (x or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    def list_html(items):
        if not items: return "<p>—</p>"
        lis = "".join(f"<li>{esc(str(i))}</li>" for i in items)
        return f"<ul>{lis}</ul>"

    summary = esc(data.get("executive_summary") or data.get("one_sentence_takeaway") or "")
    hook = esc(data.get("inspiration_hook") or "")
    why = esc(data.get("why_it_matters") or "")
    tips = data.get("study_tips") or []
    author = data.get("about_author") or {}
    quotes = data.get("quote_bank") or []
    themes = data.get("themes_detailed") or []

    theme_rows = ""
    for t in themes:
        theme_rows += f"<tr><td>{esc(t.get('theme',''))}</td><td>{esc(t.get('explanation',''))}</td><td>{esc(' | '.join(t.get('evidence_quotes',[]) or []))}</td></tr>"

    html = f"""
<!doctype html><html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Literature Insight — {esc(category.title())}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;line-height:1.5;margin:24px;color:#0f172a}}
h1,h2,h3{{color:#0f172a;margin:0.4em 0}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;background:#fff}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}}
.small{{color:#475569;font-size:12px}}
</style></head>
<body>
<h1>Literature Insight — {esc(category.title())}</h1>
<div class="small">Generated: {ts}</div>

<div class="card"><h2>Executive Summary</h2><p>{summary or '—'}</p></div>
<div class="card"><h2>Inspiration Hook</h2><p>{esc(hook) or '—'}</p></div>
<div class="card"><h2>Why It Matters</h2><p>{esc(why) or '—'}</p></div>

<div class="card"><h2>Study Tips</h2>{list_html(tips)}</div>
<div class="card"><h2>About the Author</h2>
<p><b>Name:</b> {esc(author.get('name','—'))}</p>
<p><b>Era/Period:</b> {esc(author.get('era_or_period','—'))}</p>
<p><b>Region:</b> {esc(author.get('nationality_or_region','—'))}</p>
<p><b>Relevance:</b> {esc(author.get('relevance_to_text','—'))}</p>
</div>

<div class="card">
<h2>Themes & Evidence</h2>
<table><thead><tr><th>Theme</th><th>Explanation</th><th>Evidence</th></tr></thead>
<tbody>{theme_rows or '<tr><td colspan="3">—</td></tr>'}</tbody></table>
</div>

<div class="card"><h2>Quote Bank</h2>{list_html(quotes)}</div>
<div class="small">© Suvichaar Literature Insight</div>
</body></html>
"""
    return html


# =========================
# UI INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a poem/play/story/essay (optional)", height=160, placeholder="e.g., Your face is like Moon")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols_top = st.columns(5)
with cols_top[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect", "English", "Hindi"], index=0)
with cols_top[1]:
    detail_level = st.slider("Detail level", 1, 5, 5, help="Higher → richer but concise output.")
with cols_top[2]:
    evidence_per_section = st.slider("Evidence/examples per list", 1, 6, 4)
with cols_top[3]:
    teacher_mode = st.toggle("Include Teacher View", value=True, help="Adds objectives, MCQs, and discussion set.")
with cols_top[4]:
    allow_padding = st.toggle("Auto-fill missing items", value=False, help="ON pads lists to the target count; OFF keeps only model output.")

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
    chip(f"Language: {explain_lang}", "#0891b2")
    st.write("")

    # Category detection
    guessed = heuristic_guess_category(source_text)
    try:
        cat = classify_with_gpt(source_text) or guessed
    except Exception:
        cat = guessed
    chip(f"Category: {cat}", "#16a34a")

    # Study template (deterministic)
    study_template = make_study_template(cat)

    # Prompt & model call
    safe_text = make_classroom_safe(source_text)
    system_msg = (
        "You are a veteran literature teacher for school students. "
        "Analyze the text in a CLASSROOM-SAFE way. Avoid explicit/graphic language. "
        "Stick to evidence from the text."
        + (" Respond in Hindi." if explain_lang.startswith("hi") else " Respond in English.")
    )
    user_msg = f"TEXT TO ANALYZE (verbatim):\n{safe_text}\n\n{build_schema_prompt(cat, explain_lang, detail_level, evidence_per_section, teacher_mode)}"

    with st.spinner("Generating structured DATA via Azure…"):
        ok, content = call_azure_chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.15 if detail_level >= 4 else 0.1,
            max_tokens=4800,
            force_json=True
        )

    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in safer mode…")
        ok, content = call_azure_chat(
            [
                {"role": "system", "content": "You are a cautious literature teacher. Avoid explicit terms; use neutral wording."},
                {"role": "user", "content": f"Return ONLY JSON with a 'data' object for category '{cat}', detail {detail_level}, ≤ {evidence_per_section} items per list, language {explain_lang}. Text:\n{safe_text}"}
            ],
            temperature=0.0, max_tokens=4400, force_json=True
        )

    if not ok:
        st.error(content)
        st.stop()

    parsed = robust_json_parse(content)
    data = parsed.get("data") if isinstance(parsed, dict) else None

    if not isinstance(data, dict) or not data:
        st.info("The model returned malformed JSON. Retrying with a minimal skeleton…")
        ok2, content2 = call_azure_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Return ONLY JSON: {{\"data\": {{\"executive_summary\":\"...\",\"about_author\":{{}},\"themes_detailed\":[],\"quote_bank\":[]}}}} for a '{cat}' text (≤ {evidence_per_section} items per list). Text:\n{safe_text}"}
            ],
            temperature=0.0, max_tokens=1200, force_json=True
        )
        parsed2 = robust_json_parse(content2) if ok2 else None
        data = parsed2.get("data") if isinstance(parsed2, dict) else None

    if not isinstance(data, dict) or not data:
        st.error("Model did not return valid JSON.")
        st.stop()

    if allow_padding:
        data = enforce_minimums(data, evidence_per_section)

    # ============= PRESENTATION =============
    st.markdown("---")
    tabs = st.tabs([
        "Study Template", "Overview", "Text Insights", "Author & Characters",
        "Themes & Quotes", "Activities & Rubrics", "Q&A + Emotional Arc",
        "Teacher View", "Export / JSON"
    ])

    # Study Template
    with tabs[0]:
        h2("Category-specific Study Template", "🧩")
        st.write(f"**Category:** {study_template['category'].title()}")
        st.markdown("**Core Sections**")
        st.write("\n".join(f"• {s}" for s in study_template["sections"]))
        if study_template.get("extras"):
            st.markdown("**Extras (focus for this category)**")
            st.write("\n".join(f"• {s}" for s in study_template["extras"]))

    # Overview
    with tabs[1]:
        h2("Executive Summary", "🧭")
        st.write(data.get("executive_summary") or data.get("one_sentence_takeaway") or "—")

        c1, c2 = st.columns(2)
        with c1:
            h2("Inspiration Hook", "✨")
            st.write(data.get("inspiration_hook") or "—")
        with c2:
            h2("Why It Matters", "🎯")
            st.write(data.get("why_it_matters") or "—")

        c3, c4 = st.columns(2)
        with c3:
            h2("Study Tips", "📝")
            st.write(safe_join_bullets(data.get("study_tips")))
        with c4:
            h2("Extension Reading", "📚")
            st.write(safe_join_bullets(data.get("extension_reading")))

    # Text Insights
    with tabs[2]:
        lcol, rcol = st.columns(2)
        with lcol:
            h2("Literal meaning", "📘")
            st.write(data.get("literal_meaning", "—"))
            h2("Figurative meaning / themes", "🌊")
            st.write(data.get("figurative_meaning", "—"))
        with rcol:
            h2("Tone & Mood", "🎼")
            st.write(data.get("tone_mood", "—"))
            h2("One-sentence takeaway", "✅")
            st.write(data.get("one_sentence_takeaway", "—"))

        if cat == "poetry":
            if data.get("structure_overview"):
                h2("Structure overview", "🧩")
                st.json(data["structure_overview"], expanded=False)
            if show_line_by_line and data.get("line_by_line"):
                h2("Line-by-line", "📖")
                for i, it in enumerate(data["line_by_line"], start=1):
                    st.markdown(f"**Line {i}:** {it.get('line','')}")
                    st.write(it.get("explanation", ""))
                    dev = it.get("device_notes", "")
                    if dev:
                        st.caption(f"Device notes: {dev}")
                    st.divider()
            if show_devices_table and data.get("devices"):
                h2("Devices", "🎭")
                st.table([
                    {"device": d.get("name", ""), "evidence": d.get("evidence", ""), "why": d.get("explanation", "")}
                    for d in data["devices"]
                ])
            if data.get("imagery_map"):
                h2("Imagery map", "🌈")
                st.table(data["imagery_map"])
            if data.get("symbol_table"):
                h2("Symbols", "🔶")
                st.table(data["symbol_table"])

        if cat in ("play", "story"):
            if show_line_by_line and cat == "play" and data.get("dialogue_beats"):
                h2("Dialogue beats", "💬")
                st.table(data["dialogue_beats"])
            if data.get("plot_points"):
                h2("Plot points", "🧭")
                st.table(data["plot_points"])
            if data.get("conflict"):
                h2("Conflict", "⚔️")
                st.write(data["conflict"])

        if cat == "essay":
            if data.get("thesis"):
                h2("Thesis", "🎯")
                st.write(data["thesis"])
            if data.get("key_points"):
                h2("Key points", "📌")
                st.table(data["key_points"])
            if data.get("rhetorical_devices"):
                h2("Rhetorical devices", "✨")
                st.table(data["rhetorical_devices"])

    # Author & Characters
    with tabs[3]:
        if show_author and data.get("about_author"):
            h2("About the Author", "✍️")
            st.json(data["about_author"], expanded=False)

        if show_characters and data.get("characters"):
            h2("Character Sketches", "👥")
            st.table([{
                "name": c.get("name", ""),
                "role": c.get("role", ""),
                "traits": ", ".join(c.get("traits", [])) if isinstance(c.get("traits"), list) else c.get("traits", ""),
                "motives": ", ".join(c.get("motives", [])) if isinstance(c.get("motives"), list) else c.get("motives", ""),
                "arc": c.get("arc_or_change", "")
            } for c in data["characters"]])
            for c in data["characters"]:
                if c.get("key_quotes"):
                    st.markdown(f"**Key quotes — {c.get('name','Character')}**")
                    st.write("\n".join(f"• {q.get('quote','')} — {q.get('explanation','')}" for q in c["key_quotes"]))

    # Themes & Quotes
    with tabs[4]:
        if data.get("themes_detailed"):
            h2("Themes & Evidence", "🧠")
            st.table([{
                "theme": t.get("theme", ""),
                "explanation": t.get("explanation", ""),
                "evidence": " | ".join(t.get("evidence_quotes", [])) if isinstance(t.get("evidence_quotes", []), list) else t.get("evidence_quotes", "")
            } for t in data["themes_detailed"]])
        if show_quotes and data.get("quote_bank"):
            h2("Quote Bank", "❝")
            st.write("\n".join(f"• {q}" for q in data["quote_bank"]))
        if data.get("comparative_texts"):
            h2("Comparative Texts", "🔁")
            st.table(data["comparative_texts"])
        if data.get("cross_curricular_links"):
            h2("Cross-curricular Links", "🔗")
            st.table(data["cross_curricular_links"])
        if data.get("adaptation_ideas"):
            h2("Adaptation Ideas", "🎬")
            st.write("\n".join(f"• {a}" for a in data["adaptation_ideas"]))

    # Activities & Rubrics
    with tabs[5]:
        if show_activities and data.get("activities"):
            acts = data["activities"]
            for section in ["pre_reading", "during_reading", "post_reading", "creative_tasks", "projects"]:
                if acts.get(section):
                    h2(section.replace("_", " ").title(), "🛠️")
                    for item in acts[section]:
                        st.markdown(f"- **{item.get('title','Activity')}**")
                        if "steps" in item:
                            for i, sstep in enumerate(item["steps"], start=1):
                                st.write(f"  {i}. {sstep}")
                        for k in ("duration_min", "strategy", "type", "deliverable", "criteria", "outcome", "prompt"):
                            if item.get(k):
                                st.caption(f"{k}: {item[k]}")
        if show_rubric and data.get("assessment_rubric"):
            h2("Assessment Rubric", "🧪")
            st.json(data["assessment_rubric"], expanded=False)

        if data.get("vocabulary_glossary"):
            h2("Glossary", "📒")
            st.table(data["vocabulary_glossary"])
        if data.get("misconceptions"):
            h2("Misconceptions to avoid", "⚠️")
            st.write("\n".join(f"• {m}" for m in data["misconceptions"]))

    # Q&A + Emotional Arc
    with tabs[6]:
        if data.get("emotional_arc"):
            h2("Emotional Arc (Build & Release)", "💓")
            st.table(data["emotional_arc"])
        if data.get("questions_short"):
            h2("Short Questions & Answers", "❔")
            st.table(data["questions_short"])
        if data.get("questions_long"):
            h2("Long Questions (Analytical) & Model Answers", "🧩")
            st.table(data["questions_long"])

    # Teacher View
    with tabs[7]:
        if teacher_mode and data.get("teacher_view"):
            tv = data["teacher_view"]
            h2("Learning Objectives", "🎓")
            st.write("\n".join(f"• {o}" for o in tv.get("learning_objectives", [])) or "—")
            h2("Discussion Questions", "💡")
            st.write("\n".join(f"• {q}" for q in tv.get("discussion_questions", [])) or "—")
            h2("Quick Assessment (MCQ)", "📝")
            if tv.get("quick_assessment_mcq"):
                st.table(tv["quick_assessment_mcq"])
            else:
                st.write("—")
        else:
            st.info("Teacher View is disabled or not available in response.")

    # Export / JSON
    with tabs[8]:
        h2("Download HTML snapshot", "⬇️")
        html_str = build_portable_html(data, cat)
        st.download_button(
            "Download portable HTML",
            data=html_str.encode("utf-8"),
            file_name=f"literature_{cat}_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
        bundle = {"category": cat, "template": study_template, "data": data}
        st.markdown("#### Raw JSON (template + data)")
        st.json(bundle, expanded=False)
        st.download_button(
            "Download JSON (template + data)",
            data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_{cat}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
