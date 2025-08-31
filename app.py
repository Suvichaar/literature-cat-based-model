# app.py
import os
import re
import json
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# --- Azure Document Intelligence (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar — Literature Insight", page_icon="📚", layout="wide")
st.title("📚 Suvichaar — Literature Insight (Templates + Q&A + Exports)")
st.caption(
    "Upload/paste text → OCR → Auto-detect category → Editable category template → "
    "Generate structured analysis, short/long answers, and quiz → Export HTML/JSON."
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
# UTILITIES
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


def call_azure_chat(messages, *, temperature=0.1, max_tokens=4000, force_json=False):
    """Azure OpenAI Chat Completions call with optional strict JSON."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}  # strict JSON output (preview)
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return True, content
        if r.status_code == 400 and "filtered" in r.text.lower():
            return False, "FILTERED"
        return False, f"Azure error {r.status_code}: {r.text[:500]}"
    except Exception as e:
        return False, f"Azure request failed: {e}"


def strip_code_fences(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    return s


def repair_json(s: str) -> str:
    """Repair common JSON issues."""
    if not s:
        return s
    s = strip_code_fences(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",(\s*[}\]])", r"\1", s)  # trailing commas
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    return s.strip()


def robust_parse(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(repair_json(s))
    except Exception:
        return None


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


def classify_with_gpt(txt: str, lang: str) -> str:
    """Ask Azure to classify; fall back to heuristics."""
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
# CATEGORY TEMPLATES (EDITABLE)
# =========================
TEMPLATES = {
    "poetry": {
        "sections": [
            "Title & Poet",
            "Speaker/Voice & POV",
            "Context/Background (era, movement)",
            "Central Idea (Gist)",
            "Structure (stanzas, rhyme scheme, meter, line length)",
            "Tone & Mood",
            "Literary & Sound Devices (with evidence)",
            "Imagery & Symbolism",
            "Line-by-Line/Section-wise Explanation",
            "Themes & Messages",
            "Important Quotes (key lines + meaning)",
            "Vocabulary & Meanings",
            "Comparative Texts (optional)",
            "Emotional Arc (setup → tension → turn → release)",
            "Study Tips",
            "Extension Reading",
            "Activities (pre/during/post), Creative Tasks, Projects",
            "Assessment Rubric",
            "About the Poet"
        ]
    },
    "play": {
        "sections": [
            "Title & Playwright",
            "Dramatic Context (act/scene, setting/time)",
            "Characters & Relationships (brief sketches)",
            "Plot Overview (scene/act-wise)",
            "Conflict (internal/external)",
            "Dialogue Beats (key exchanges & function)",
            "Stage Directions & Performance Notes",
            "Dramatic Devices (soliloquy, aside, irony, foreshadowing)",
            "Themes & Messages",
            "Tone & Mood / Atmosphere",
            "Important Quotes (with function)",
            "Vocabulary & Meanings",
            "Comparative Productions (optional)",
            "Study Tips",
            "Extension Reading",
            "Activities (table-read, blocking, radio play), Projects",
            "Assessment Rubric",
            "About the Playwright"
        ]
    },
    "story": {
        "sections": [
            "Title & Author",
            "Narrative Voice & Focalization",
            "Setting (time/place/social context)",
            "Characters (sketches)",
            "Plot Points (exposition → rising → climax → falling → resolution)",
            "Conflict (type & stakes)",
            "Themes & Messages",
            "Symbols & Motifs",
            "Tone & Style",
            "Important Quotes (evidence)",
            "Vocabulary & Meanings",
            "Comparative Texts (optional)",
            "Study Tips",
            "Extension Reading",
            "Activities (storyboard, comic strip, podcast), Projects",
            "Assessment Rubric",
            "About the Author"
        ]
    },
    "essay": {
        "sections": [
            "Title & Author",
            "Context/Purpose/Audience",
            "Thesis/Main Claim",
            "Key Points & Evidence (with counterpoints if any)",
            "Structure (intro/body/conclusion)",
            "Rhetorical Devices & Techniques",
            "Tone & Register",
            "Themes/Ideas",
            "Important Quotes/Examples",
            "Vocabulary & Meanings (academic terms)",
            "Comparative Readings (optional)",
            "Study Tips",
            "Extension Reading",
            "Activities (debate, op-ed writing), Projects",
            "Assessment Rubric",
            "About the Author"
        ]
    },
    "biography": {
        "sections": [
            "Title & Biographical Subject",
            "Author (biographer) & Perspective",
            "Context/Scope (period covered)",
            "Timeline (key life events)",
            "Qualities/Values of the Subject",
            "Contributions/Impact/Legacy",
            "Themes/Morals",
            "Tone & Approach (objective/celebratory/critical)",
            "Important Quotes/Incidents",
            "Vocabulary & Meanings (historical terms)",
            "Study Tips",
            "Extension Reading",
            "Activities (timeline poster, role-play interview), Projects",
            "Assessment Rubric",
            "About the Biographer"
        ]
    },
    "autobiography": {
        "sections": [
            "Title & Author (the subject)",
            "Context/Period of Life Covered",
            "Narrative Voice & Reliability",
            "Episodes/Incidents & Reflections",
            "Lessons/Life Themes",
            "Tone & Style",
            "Important Quotes",
            "Vocabulary & Meanings",
            "Study Tips",
            "Extension Reading",
            "Activities (memoir page, diary reconstruction), Projects",
            "Assessment Rubric",
            "About the Author"
        ]
    },
    "speech": {
        "sections": [
            "Title & Speaker",
            "Occasion/Context/Audience",
            "Purpose (inform/persuade/inspire)",
            "Main Claims & Key Points",
            "Rhetorical Devices (repetition, anaphora, parallelism, allusion, rhetorical questions)",
            "Structure (opening → body → close/call-to-action)",
            "Tone & Register",
            "Evidence/Appeals (ethos/pathos/logos)",
            "Memorable Lines/Quotes",
            "Vocabulary & Meanings",
            "Study Tips",
            "Extension Reading/Viewing",
            "Activities (delivery practice, speech rewrite), Projects",
            "Assessment Rubric",
            "About the Speaker"
        ]
    },
    "letter": {
        "sections": [
            "Type (formal/informal)",
            "Context/Purpose & Audience",
            "Format (addresses, date, salutation, closing)",
            "Body Points (reasoning/examples)",
            "Tone & Register",
            "Key Lines (requests/assurances)",
            "Vocabulary & Phrases (polite/requesting)",
            "Study Tips",
            "Activities (drafting/editing), Projects",
            "Assessment Rubric"
        ]
    },
    "diary": {
        "sections": [
            "Date/Time & Context",
            "Events (what happened)",
            "Feelings/Reflections",
            "Lessons/Insights",
            "Tone & Voice",
            "Key Lines",
            "Vocabulary & Expressions",
            "Study Tips",
            "Activities (journaling prompts), Projects",
            "Assessment Rubric"
        ]
    },
    "report": {
        "sections": [
            "Topic/Title",
            "Purpose & Audience",
            "Sections (Introduction, Method, Observation/Data, Discussion, Conclusion)",
            "Findings (bullet points)",
            "Recommendations/Next Steps",
            "Tone & Register (objective/concise)",
            "Figures/Tables (if any)",
            "Vocabulary (technical terms)",
            "Study Tips",
            "Activities (mini-investigation), Projects",
            "Assessment Rubric"
        ]
    },
    "folk_tale": {
        "sections": [
            "Title & Cultural Origin",
            "Characters (types/roles)",
            "Setting",
            "Plot Outline (pattern/repetition)",
            "Motifs/Symbols",
            "Moral/Lesson",
            "Themes & Cultural Values",
            "Tone & Style (oral features)",
            "Key Lines/Formulaic Openings",
            "Vocabulary (culture-specific terms)",
            "Study Tips",
            "Activities (retelling, skit), Projects",
            "Assessment Rubric"
        ]
    },
    "myth": {
        "sections": [
            "Title & Tradition/Origin",
            "Deities/Beings & Symbols",
            "Cosmogony/Phenomenon Explained",
            "Plot Outline",
            "Themes (hubris, fate, order vs. chaos)",
            "Cultural Values/Functions",
            "Tone & Style",
            "Key Lines/Symbolic Moments",
            "Vocabulary (mythic terms)",
            "Study Tips",
            "Activities (symbol map, modern retelling), Projects",
            "Assessment Rubric"
        ]
    },
    "legend": {
        "sections": [
            "Title & Historical Backdrop",
            "Central Figure/Hero",
            "Notable Events/Feats",
            "Elements of Fact vs. Embellishment",
            "Themes (bravery, loyalty, identity)",
            "Cultural/Historical Significance",
            "Tone & Style",
            "Key Lines",
            "Vocabulary",
            "Study Tips",
            "Activities (timeline vs. sources), Projects",
            "Assessment Rubric"
        ]
    }
}


# =========================
# PROMPT BUILDERS (MORE DETAILED)
# =========================
def build_analysis_prompt(category: str, language_code: str, detail: int, evidence_cap: int, selected_sections: list) -> str:
    """
    Ask the model to fill a compact 'data' object aligned to the chosen sections,
    with a *bit more detail* than before. We softly target 4–6 items for key lists.
    """
    # try to hit a richer target within caps
    target_min = max(3, evidence_cap)             # minimum items per list we try to cover
    target_max = max(target_min + 1, evidence_cap + 2)  # soft upper bound (model may return up to cap)

    extra_keys = []
    if category == "poetry":
        extra_keys = ["devices", "imagery_map", "symbol_table", "line_by_line", "structure_overview", "speaker_or_voice"]
    elif category in ("play", "story"):
        extra_keys = ["characters", "plot_points", ("dialogue_beats" if category == "play" else "")]
    elif category == "essay":
        extra_keys = ["thesis", "key_points", "rhetorical_devices"]

    schema_hint = {
        "executive_summary": "6–8 lines overview (concise sentences)",
        "inspiration_hook": "curiosity hook (1–2 lines)",
        "why_it_matters": "transferable skills/values (2–3 lines)",
        "study_tips": [f"{target_min}–{min(6, target_max)} concise tips"],
        "extension_reading": [f"{target_min}–{min(6, target_max)} related texts"],
        "themes_detailed": [f"{target_min}–{min(6, target_max)} themes with 1–2 evidence quotes each"],
        "quote_bank": [f"{target_min}–{min(6, target_max)} key lines (verbatim)"]
    }

    return (
        "You are an expert literature teacher. Analyze the text below in a CLASSROOM-SAFE way.\n"
        "Return ONLY JSON with a single key 'data'. No markdown, no comments.\n"
        f"Language: {language_code}\n"
        f"Category: {category}\n"
        f"Detail level (1–5): {detail} (aim for rich but concise coverage)\n"
        f"For list fields, aim for {target_min}–{min(6, target_max)} items, but DO NOT exceed {evidence_cap} per list.\n"
        "Required keys (as supported by text): executive_summary, inspiration_hook, why_it_matters, "
        "study_tips, extension_reading, themes_detailed, quote_bank, tone_mood.\n"
        "If available from this category, also include: "
        + ", ".join([k for k in extra_keys if k]) + ".\n"
        "Additionally include: literal_meaning (1–3 lines), figurative_meaning (1–3 lines), "
        "one_sentence_takeaway (single line), emotional_arc (3–5 beats with evidence).\n"
        "Align content to these teacher sections (titles to guide organization):\n"
        + json.dumps(selected_sections, ensure_ascii=False)
        + "\nSchema hints:\n" + json.dumps(schema_hint, ensure_ascii=False, indent=2)
    )


def build_qa_prompt(kind: str, count: int, language_code: str, difficulty: str = "balanced") -> str:
    """
    kind: 'short', 'long', or 'quiz'
    """
    if kind == "short":
        return (
            "Create SHORT-ANSWER Q&A for the given text. "
            f"Return ONLY JSON with key 'questions_short' as an array of exactly {count} objects {{q,a}}.\n"
            "Answers must be 2–3 lines and include a brief textual cue or short quote when helpful. "
            "Classroom-safe phrasing only."
            f"\nLanguage: {language_code}. Difficulty: {difficulty}."
        )
    if kind == "long":
        return (
            "Create ANALYTICAL LONG-ANSWER Q&A for the given text. "
            f"Return ONLY JSON with key 'questions_long' as an array of exactly {count} objects {{q,a}}.\n"
            "Each answer should be 6–10 lines, well-structured, and include 1–2 brief quotes as evidence. "
            "Prefer depth and clear reasoning."
            f"\nLanguage: {language_code}. Difficulty: {difficulty}."
        )
    return (
        "Create MULTIPLE-CHOICE questions (MCQs) for the given text. "
        f"Return ONLY JSON with key 'quiz' as an array of exactly {count} objects "
        "{{q, choices:[A,B,C,D], answer}}.\n"
        "One unambiguous correct answer; choices plausible and distinct; keep stems concise."
        f"\nLanguage: {language_code}. Difficulty: {difficulty}."
    )


# =========================
# PORTABLE HTML EXPORT
# =========================
def build_portable_html(bundle: dict) -> str:
    """Portable HTML snapshot: template + data + QAs + quiz."""
    cat = (bundle.get("category") or "").title()
    data = bundle.get("data") or {}
    qshort = bundle.get("questions_short") or []
    qlong  = bundle.get("questions_long") or []
    quiz   = bundle.get("quiz") or []
    template = bundle.get("template") or {}
    sections = template.get("sections", [])
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def esc(x):
        return (str(x) if x is not None else "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def list_html(items):
        if not items: return "<p>—</p>"
        lis = "".join(f"<li>{esc(i)}</li>" for i in items)
        return f"<ul>{lis}</ul>"

    # pieces
    summary = esc(data.get("executive_summary") or data.get("one_sentence_takeaway") or "")
    hook = esc(data.get("inspiration_hook") or "")
    why  = esc(data.get("why_it_matters") or "")
    tips = data.get("study_tips") or []
    ext  = data.get("extension_reading") or []
    themes = data.get("themes_detailed") or []
    quotes = data.get("quote_bank") or []
    author = data.get("about_author") or {}
    characters = data.get("characters") or []
    devices = data.get("devices") or []
    imagery = data.get("imagery_map") or []
    symbols = data.get("symbol_table") or []
    line_by_line = data.get("line_by_line") or []
    plot_points = data.get("plot_points") or []
    dialogue_beats = data.get("dialogue_beats") or []

    theme_rows = ""
    for t in themes:
        theme_rows += f"<tr><td>{esc(t.get('theme',''))}</td><td>{esc(t.get('explanation',''))}</td><td>{esc(' | '.join(t.get('evidence_quotes',[]) or []))}</td></tr>"

    char_rows = ""
    for c in characters:
        traits = ", ".join(c.get('traits',[]) or [])
        motives = ", ".join(c.get('motives',[]) or [])
        char_rows += f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('role',''))}</td><td>{esc(traits)}</td><td>{esc(motives)}</td><td>{esc(c.get('arc_or_change',''))}</td></tr>"

    device_rows = ""
    for d in devices:
        device_rows += f"<tr><td>{esc(d.get('name',''))}</td><td>{esc(d.get('evidence',''))}</td><td>{esc(d.get('explanation',''))}</td></tr>"

    plot_rows = ""
    for p in plot_points:
        plot_rows += f"<tr><td>{esc(p.get('stage',''))}</td><td>{esc(p.get('what_happens',''))}</td><td>{esc(p.get('evidence',''))}</td></tr>"

    beats_rows = ""
    for b in dialogue_beats:
        beats_rows += f"<tr><td>{esc(b.get('speaker',''))}</td><td>{esc(b.get('line',''))}</td><td>{esc(b.get('note',''))}</td></tr>"

    qshort_rows = ""
    for qa in qshort:
        qshort_rows += f"<tr><td>{esc(qa.get('q',''))}</td><td>{esc(qa.get('a',''))}</td></tr>"

    qlong_rows = ""
    for qa in qlong:
        qlong_rows += f"<tr><td>{esc(qa.get('q',''))}</td><td>{esc(qa.get('a',''))}</td></tr>"

    quiz_rows = ""
    for q in quiz:
        ch = q.get("choices") or []
        ch_s = " | ".join([esc(x) for x in ch])
        quiz_rows += f"<tr><td>{esc(q.get('q',''))}</td><td>{ch_s}</td><td>{esc(q.get('answer',''))}</td></tr>"

    html = f"""
<!doctype html><html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Literature Insight — {esc(cat)}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;line-height:1.5;margin:24px;color:#0f172a}}
h1,h2,h3{{color:#0f172a;margin:0.4em 0}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;background:#fff}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}}
.small{{color:#475569;font-size:12px}}
.kbd{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;background:#f1f5f9;border-radius:6px;padding:2px 6px}}
</style></head>
<body>
<h1>Literature Insight — {esc(cat)}</h1>
<div class="small">Generated: {ts}</div>

<div class="card"><h2>Executive Summary</h2><p>{summary or '—'}</p></div>
<div class="card"><h2>Inspiration Hook</h2><p>{hook or '—'}</p></div>
<div class="card"><h2>Why It Matters</h2><p>{why or '—'}</p></div>

<div class="card"><h2>Study Tips</h2>{list_html(tips)}</div>
<div class="card"><h2>Extension Reading</h2>{list_html(ext)}</div>

<div class="card">
<h2>Themes & Evidence</h2>
<table><thead><tr><th>Theme</th><th>Explanation</th><th>Evidence</th></tr></thead>
<tbody>{theme_rows or '<tr><td colspan="3">—</td></tr>'}</tbody></table>
</div>

<div class="card"><h2>Quote Bank</h2>{list_html(quotes)}</div>

{"<div class='card'><h2>Characters</h2><table><thead><tr><th>Name</th><th>Role</th><th>Traits</th><th>Motives</th><th>Arc/Change</th></tr></thead><tbody>"+(char_rows or '<tr><td colspan=\"5\">—</td></tr>')+"</tbody></table></div>" if characters else ""}

{"<div class='card'><h2>Devices</h2><table><thead><tr><th>Device</th><th>Evidence</th><th>Why</th></tr></thead><tbody>"+(device_rows or '<tr><td colspan=\"3\">—</td></tr>')+"</tbody></table></div>" if devices else ""}

{"<div class='card'><h2>Plot Points</h2><table><thead><tr><th>Stage</th><th>What Happens</th><th>Evidence</th></tr></thead><tbody>"+(plot_rows or '<tr><td colspan=\"3\">—</td></tr>')+"</tbody></table></div>" if plot_points else ""}

{"<div class='card'><h2>Dialogue Beats</h2><table><thead><tr><th>Speaker</th><th>Line</th><th>Note</th></tr></thead><tbody>"+(beats_rows or '<tr><td colspan=\"3\">—</td></tr>')+"</tbody></table></div>" if dialogue_beats else ""}

<div class="card">
<h2>Short Answers</h2>
<table><thead><tr><th>Question</th><th>Answer</th></tr></thead>
<tbody>{qshort_rows or '<tr><td colspan="2">—</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Long Answers</h2>
<table><thead><tr><th>Question</th><th>Answer</th></tr></thead>
<tbody>{qlong_rows or '<tr><td colspan="2">—</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Quiz (MCQ)</h2>
<table><thead><tr><th>Question</th><th>Choices</th><th>Answer</th></tr></thead>
<tbody>{quiz_rows or '<tr><td colspan="3">—</td></tr>'}</tbody></table>
</div>

<div class="card"><h2>Template Sections</h2>{list_html(sections)}</div>

<div class="small">© Suvichaar Literature Insight</div>
</body></html>
"""
    return html


# =========================
# UI — INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste text (optional)", height=160, placeholder="Paste poem/play/story/essay here")
files = st.file_uploader("Or upload an image/PDF", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols = st.columns(6)
with cols[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols[1]:
    detail_level = st.slider("Detail level", 1, 5, 5)  # nudge to 5 for richer data
with cols[2]:
    evidence_cap = st.slider("Max items per list", 2, 8, 5)  # allow richer lists
with cols[3]:
    gen_short = st.checkbox("Gen Short Answers", value=True)
with cols[4]:
    gen_long = st.checkbox("Gen Long Answers", value=True)
with cols[5]:
    gen_quiz = st.checkbox("Gen Quiz (MCQ)", value=True)

cols2 = st.columns(2)
with cols2[0]:
    quiz_count = st.number_input("Quiz count", min_value=6, max_value=20, value=10, step=1)
with cols2[1]:
    difficulty = st.selectbox("Difficulty", ["easy", "balanced", "challenging"], index=1)

run = st.button("🔎 Analyze & Open Template")


# =========================
# SESSION STATE
# =========================
for key in ["template_df", "category", "source_text", "data", "questions_short", "questions_long", "quiz", "template_used"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# MAIN
# =========================
if run:
    # 1) Build source text
    source_text = (text_input or "").strip()
    if files and not source_text:
        with st.spinner("Running OCR…"):
            blob = files.read()
            ocr_text = ocr_read_any(blob)
            if ocr_text:
                source_text = ocr_text
                st.success("OCR extracted text.")
                with st.expander("Show OCR text"):
                    st.write(ocr_text[:20000])
            else:
                st.error("OCR returned no text. Try a clearer image or paste text.")
                st.stop()

    if not source_text:
        st.error("Please paste text or upload a file.")
        st.stop()

    st.session_state.source_text = source_text

    # 2) Language
    detected = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else "hi" if lang_choice == "Hindi" else detected

    # 3) Category detection
    guessed = heuristic_guess_category(source_text)
    try:
        cat = classify_with_gpt(source_text, explain_lang) or guessed
    except Exception:
        cat = guessed

    st.session_state.category = cat

    # 4) Load category template into editable grid
    template = TEMPLATES.get(cat, TEMPLATES["story"])
    sections = template.get("sections", [])
    df = pd.DataFrame({"include": [True]*len(sections), "section": sections})
    st.session_state.template_df = df
    st.session_state.template_used = {"sections": sections}

    st.success(f"Language: {explain_lang}  •  Category detected: {cat}")


# If we have a template ready, open the editing interface
if st.session_state.template_df is not None and st.session_state.category is not None:
    st.markdown("---")
    st.subheader("🧩 Category Template (Editable)")
    st.caption("Toggle sections on/off, reorder rows, or rename sections as you like.")
    edited_df = st.data_editor(
        st.session_state.template_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    st.session_state.template_df = edited_df

    # Selected sections (in order, included only)
    selected_sections = [r["section"] for _, r in edited_df.iterrows() if bool(r["include"]) and str(r["section"]).strip()]
    if not selected_sections:
        st.info("Select at least one section to guide the generator.")

    # Controls for generation
    st.markdown("### ⚙️ Generation Controls")
    colg = st.columns(4)
    with colg[0]:
        regen_analysis = st.button("🧭 Generate Structured Analysis")
    with colg[1]:
        make_short = st.button("✍️ Generate Short Answers") if gen_short else None
    with colg[2]:
        make_long = st.button("🧠 Generate Long Answers") if gen_long else None
    with colg[3]:
        make_quiz = st.button("📝 Generate Quiz") if gen_quiz else None

    # Prepare common context
    explain_lang = detect_hi_or_en(st.session_state.source_text) if lang_choice == "Auto-detect" else ("en" if lang_choice == "English" else "hi")
    safe_text = make_classroom_safe(st.session_state.source_text)
    category = st.session_state.category

    # SYSTEM message
    sys_msg = (
        "You are a veteran literature teacher for school students. "
        "Analyze the text in a CLASSROOM-SAFE way. Avoid explicit/graphic language. "
        "Stick to evidence from the text."
        + (" Respond in Hindi." if explain_lang.startswith("hi") else " Respond in English.")
    )

    # 1) Structured Analysis (more detailed)
    if regen_analysis:
        with st.spinner("Generating structured analysis…"):
            prompt = build_analysis_prompt(category, explain_lang, detail_level, evidence_cap, selected_sections)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.18 if detail_level >= 4 else 0.12,
                max_tokens=3600,
                force_json=True
            )
        if not ok and content == "FILTERED":
            st.warning("Sensitive content detected. Retrying with tighter safe mode…")
            ok, content = call_azure_chat(
                [{"role": "system", "content": "You are a cautious school literature teacher. Avoid explicit terms; use neutral wording."},
                 {"role": "user", "content": f"Return ONLY JSON with 'data' for category '{category}'. "
                                              f"Keep arrays ≤ {evidence_cap}. Language {explain_lang}.\nTEXT:\n{safe_text}"}],
                temperature=0.0, max_tokens=3000, force_json=True
            )

        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if not isinstance(data, dict):
                st.error("Model didn't return valid JSON with a top-level 'data' object.")
            else:
                st.session_state.data = data
                st.markdown("#### 📦 Structured Analysis (JSON)")
                st.json(data, expanded=False)

    # 2) Short Answers
    if gen_short and make_short:
        with st.spinner("Generating short answers…"):
            # default to 5 if user set cap small
            short_n = max(5, int(evidence_cap))
            prompt = build_qa_prompt("short", short_n, explain_lang, difficulty=difficulty)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.12, max_tokens=2200, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("questions_short")
            if not isinstance(arr, list):
                st.error("Expected key 'questions_short' as an array.")
            else:
                st.session_state.questions_short = arr
                st.markdown("#### ❔ Short Q&A")
                st.table(arr)

    # 3) Long Answers
    if gen_long and make_long:
        with st.spinner("Generating long answers…"):
            long_n = max(5, int(evidence_cap))
            prompt = build_qa_prompt("long", long_n, explain_lang, difficulty=difficulty)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.16, max_tokens=3000, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("questions_long")
            if not isinstance(arr, list):
                st.error("Expected key 'questions_long' as an array.")
            else:
                st.session_state.questions_long = arr
                st.markdown("#### 🧩 Long Q&A (Analytical)")
                st.table(arr)

    # 4) Quiz
    if gen_quiz and make_quiz:
        with st.spinner("Generating quiz…"):
            prompt = build_qa_prompt("quiz", int(quiz_count), explain_lang, difficulty=difficulty)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.15, max_tokens=2600, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("quiz")
            if not isinstance(arr, list):
                st.error("Expected key 'quiz' as an array.")
            else:
                st.session_state.quiz = arr
                st.markdown("#### 📝 Quiz (MCQ)")
                st.table(arr)

    # ============= PRESENTATION LAYOUT =============
    st.markdown("---")
    tabs = st.tabs([
        "Overview", "Text Insights", "Characters / Devices", "Themes & Quotes",
        "Q&A", "Quiz", "Export"
    ])

    data = st.session_state.data or {}

    # ----- Overview -----
    with tabs[0]:
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
            tips = data.get("study_tips") or []
            st.write("\n".join(f"• {x}" for x in tips) or "—")
        with c4:
            st.markdown("### 📚 Extension Reading")
            ext = data.get("extension_reading") or []
            st.write("\n".join(f"• {x}" for x in ext) or "—")

    # ----- Text Insights -----
    with tabs[1]:
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("### 📘 Literal Meaning")
            st.write(data.get("literal_meaning", "—"))
            st.markdown("### 🌊 Figurative Meaning / Themes")
            st.write(data.get("figurative_meaning", "—"))
        with rcol:
            st.markdown("### 🎼 Tone & Mood")
            st.write(data.get("tone_mood", "—"))
            st.markdown("### ✅ One-sentence Takeaway")
            st.write(data.get("one_sentence_takeaway", "—"))

        # Poetry-specific
        if st.session_state.category == "poetry":
            if data.get("structure_overview"):
                st.markdown("### 🧩 Structure Overview")
                st.json(data["structure_overview"], expanded=False)
            if data.get("line_by_line"):
                st.markdown("### 📖 Line-by-Line / Section-wise")
                st.table(data["line_by_line"])

    # ----- Characters / Devices -----
    with tabs[2]:
        if st.session_state.category in ("play", "story"):
            if data.get("characters"):
                st.markdown("### 👥 Characters")
                # Show a tidy character table
                rows = []
                for c in data["characters"]:
                    rows.append({
                        "name": c.get("name",""),
                        "role": c.get("role",""),
                        "traits": ", ".join(c.get("traits",[]) or []),
                        "motives": ", ".join(c.get("motives",[]) or []),
                        "arc": c.get("arc_or_change","")
                    })
                st.table(rows)
            if st.session_state.category == "play" and data.get("dialogue_beats"):
                st.markdown("### 💬 Dialogue Beats")
                st.table(data["dialogue_beats"])
            if data.get("plot_points"):
                st.markdown("### 🧭 Plot Points")
                st.table(data["plot_points"])

        if st.session_state.category == "poetry":
            if data.get("devices"):
                st.markdown("### 🎭 Devices (Poetic/Literary/Sound)")
                st.table([{
                    "device": d.get("name",""),
                    "evidence": d.get("evidence",""),
                    "why": d.get("explanation","")
                } for d in data["devices"]])
            if data.get("imagery_map"):
                st.markdown("### 🌈 Imagery Map")
                st.table(data["imagery_map"])
            if data.get("symbol_table"):
                st.markdown("### 🔶 Symbols")
                st.table(data["symbol_table"])

    # ----- Themes & Quotes -----
    with tabs[3]:
        if data.get("themes_detailed"):
            st.markdown("### 🧠 Themes & Evidence")
            st.table([{
                "theme": t.get("theme",""),
                "explanation": t.get("explanation",""),
                "evidence": " | ".join(t.get("evidence_quotes",[]) or [])
            } for t in data["themes_detailed"]])
        if data.get("quote_bank"):
            st.markdown("### ❝ Quote Bank")
            st.write("\n".join(f"• {q}" for q in data["quote_bank"]))

    # ----- Q&A -----
    with tabs[4]:
        if st.session_state.questions_short:
            st.markdown("### ❔ Short Answers")
            st.table(st.session_state.questions_short)
        if st.session_state.questions_long:
            st.markdown("### 🧩 Long Answers (Analytical)")
            st.table(st.session_state.questions_long)

    # ----- Quiz -----
    with tabs[5]:
        if st.session_state.quiz:
            st.markdown("### 📝 Quiz (MCQ)")
            st.table(st.session_state.quiz)

    # ----- Export -----
    with tabs[6]:
        st.markdown("### ⬇️ Export")
        # Build bundle
        bundle = {
            "category": st.session_state.category,
            "template": {"sections": selected_sections or (st.session_state.template_used or {}).get("sections", [])},
            "data": st.session_state.data or {},
            "questions_short": st.session_state.questions_short or [],
            "questions_long": st.session_state.questions_long or [],
            "quiz": st.session_state.quiz or []
        }

        # JSON bundle
        json_bytes = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Download JSON (template + data + Q&A + quiz)",
            data=json_bytes,
            file_name=f"literature_{st.session_state.category}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        # HTML snapshot
        html_str = build_portable_html(bundle)
        st.download_button(
            "Download Portable HTML",
            data=html_str.encode("utf-8"),
            file_name=f"literature_{st.session_state.category}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
