import os
import re
import json
from datetime import datetime
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
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
st.set_page_config(page_title="Suvichaar Literature Insight (Detailed)", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Detailed)")
st.caption("Upload a quote/poem/story/play image or paste text → OCR → Auto-detect category → Detailed, classroom-safe JSON.")

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
    """Replace risky words with classroom-friendly ones to reduce filter blocks."""
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
        r"\bsex(ual|ually)?\b": "personal topic"
    }
    for pat, sub in replacements.items():
        text = re.sub(pat, sub, text, flags=re.IGNORECASE)
    return text

# =========================
# AZURE GPT CALL
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=2600, force_json=False):
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
        return False, f"Azure error {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, f"Azure request failed: {e}"

def robust_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
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
        client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT.rstrip("/"), credential=AzureKeyCredential(AZURE_DI_KEY))
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
# CATEGORY DETECTION
# =========================
CATEGORIES = [
    "poetry", "play", "story", "essay", "biography", "autobiography",
    "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"
]

def heuristic_guess_category(txt: str) -> str:
    t = txt.strip()
    lower = t.lower()
    # quick signals
    if re.search(r'^\s*[A-Z].+\n[A-Z].+', t) and re.search(r'\n[A-Z][a-z]+:', t):  # NAME: dialogue
        return "play"
    if re.search(r'^[^\n]{0,80}\n[^\n]{0,80}\n[^\n]{0,80}(\n|$)', t) and re.search(r'[,.!?]\s*$', t) is None:
        # short lines without heavy punctuation often indicate verse
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
    safe = make_classroom_safe(txt)
    sys = "You are a precise classifier for school literature. Return ONLY one lowercase label from this set: poetry, play, story, essay, biography, autobiography, speech, letter, diary, report, folk_tale, myth, legend."
    user = f"Text:\n{safe}\n\nLabel only."
    ok, out = call_azure_chat(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.0, max_tokens=10, force_json=False
    )
    if not ok:
        return heuristic_guess_category(txt)
    label = out.strip().lower()
    label = re.sub(r'[^a-z_]', '', label)
    return label if label in CATEGORIES else heuristic_guess_category(txt)

# =========================
# DETAILED SCHEMAS
# =========================
BASE_SAFE_FIELDS = {
    "language": "en|hi",
    "text_type": "category label",
    "literal_meaning": "plain-language meaning",
    "figurative_meaning": "themes/symbolism if any",
    "tone_mood": "tone & mood words",
    "one_sentence_takeaway": "classroom-safe summary"
}

SCHEMAS = {
    "poetry": {
        **BASE_SAFE_FIELDS,
        "speaker_or_voice": "who speaks / perspective",
        "structure_overview": {"stanzas": "count", "approx_line_count": "number", "rhyme_scheme": "e.g., ABAB", "meter_or_rhythm": "if notable"},
        "devices": [
            {"name":"Simile|Metaphor|Personification|Alliteration|Assonance|Consonance|Imagery|Symbolism|Hyperbole|Enjambment|Rhyme",
             "evidence":"quoted words","explanation":"why it fits"}
        ],
        "imagery_map":[{"sense":"visual|auditory|tactile|gustatory|olfactory","evidence":"quote","effect":"reader impact"}],
        "symbol_table":[{"symbol":"...","meaning":"...","evidence":"..."}],
        "line_by_line":[{"line":"original line","explanation":"meaning","device_notes":"optional"}],
        "themes":["..."],
        "context_or_background":"poet/era/culture if relevant",
        "vocabulary_glossary":[{"term":"...","meaning":"..."}],
        "misconceptions":["..."]
    },
    "play": {
        **BASE_SAFE_FIELDS,
        "characters":[{"name":"...","trait":"...","role":"protagonist/antagonist/support"}],
        "setting":"time/place",
        "conflict":"internal/external and description",
        "dialogue_beats":[{"speaker":"Name","line":"quoted dialogue","note":"function of line (advance plot, reveal trait)"}],
        "stage_directions":"if any (short)",
        "themes":["..."]
    },
    "story": {
        **BASE_SAFE_FIELDS,
        "narrative_voice":"first/third/omniscient/limited",
        "setting":"time/place",
        "characters":[{"name":"...","trait":"...","arc":"change or static"}],
        "plot_points":[{"stage":"exposition|rising|climax|falling|resolution","what_happens":"...","evidence":"quote/line"}],
        "conflict":"type + description",
        "themes":["..."],
        "moral_or_message":"if any"
    },
    "essay": {
        **BASE_SAFE_FIELDS,
        "thesis":"author's main claim",
        "key_points":[{"point":"...", "evidence_or_example":"...", "counterpoint_if_any":"optional"}],
        "structure":"intro/body/conclusion notes",
        "tone_register":"formal/informal/analytical",
        "rhetorical_devices":[{"name":"analogy|contrast|examples|statistics","evidence":"..."}]
    },
    "biography": {
        **BASE_SAFE_FIELDS,
        "subject":"person",
        "timeline":[{"year_or_age":"...", "event":"...", "impact":"..."}],
        "qualities":["..."],
        "influence_or_impact":"...",
        "notable_works_or_contributions":["..."]
    },
    "autobiography": {
        **BASE_SAFE_FIELDS,
        "author":"person",
        "episodes":[{"when":"...", "event":"...", "reflection":"...", "lesson":"..."}],
        "themes":["..."],
        "voice_and_style":"..."
    },
    "speech": {
        **BASE_SAFE_FIELDS,
        "audience":"who",
        "purpose":"inform/persuade/inspire",
        "key_points":["..."],
        "rhetorical_devices":[{"name":"repetition|anaphora|rhetorical_question|parallelism|allusion","evidence":"...","effect":"..."}],
        "call_to_action":"if any"
    },
    "letter": {
        **BASE_SAFE_FIELDS,
        "letter_type":"formal|informal",
        "salutation":"...",
        "body_points":[{"point":"...", "example_or_reason":"..."}],
        "closing":"...",
        "tone_register":"polite/warm/requesting/complaint"
    },
    "diary": {
        **BASE_SAFE_FIELDS,
        "date_or_time_hint":"if present",
        "events":["..."],
        "feelings":"emotion words",
        "reflection":"what was learned"
    },
    "report": {
        **BASE_SAFE_FIELDS,
        "topic":"...",
        "sections":[{"heading":"Introduction|Method|Observation|Discussion|Conclusion","summary":"..."}],
        "findings":["..."],
        "recommendations":["..."]
    },
    "folk_tale": {
        **BASE_SAFE_FIELDS,
        "characters":["..."],
        "setting":"...",
        "plot_outline":["..."],
        "repeating_patterns_or_motifs":["..."],
        "moral_or_lesson":"..."
    },
    "myth": {
        **BASE_SAFE_FIELDS,
        "deities_or_symbols":["..."],
        "origin_or_explanation":"what it explains",
        "plot_outline":["..."]
    },
    "legend": {
        **BASE_SAFE_FIELDS,
        "hero_or_figure":"...",
        "historical_backdrop":"...",
        "notable_events":["..."]
    }
}

# Teacher view (optional extras)
TEACHER_EXTRAS = {
    "learning_objectives": ["..."],
    "discussion_questions": ["..."],
    "classroom_activity": [{"title":"...", "steps":["...", "..."], "duration_min": 10}],
    "quick_assessment_mcq": [{"q":"...", "choices":["A","B","C","D"], "answer":"A"}]
}

def build_schema_prompt(category: str, language_code: str, detail: int, evidence_count: int, teacher_mode: bool) -> str:
    schema = dict(SCHEMAS.get(category, SCHEMAS["story"]))  # copy
    if teacher_mode:
        schema["teacher_view"] = TEACHER_EXTRAS
    return (
        "Return ONLY a JSON object (no prose). "
        "Keys in English; values in the target explanation language. "
        "Be concise but **detailed** per the requested level; quote evidence verbatim.\n\n"
        f"Target language: {language_code}\n"
        f"Detected category: {category}\n"
        f"Detail level (1-5): {detail}  — Higher = more items, richer explanations.\n"
        f"Target evidence/examples per major section: ~{evidence_count}\n"
        "When something is not present in the text, omit that key.\n"
        "Schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
    )

# =========================
# UI INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a poem/play/story/essay (optional)", height=160, placeholder="e.g., Your face is like Moon")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols_top = st.columns(4)
with cols_top[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols_top[1]:
    detail_level = st.slider("Detail level", 1, 5, 4, help="Controls depth & number of bullets/tables returned.")
with cols_top[2]:
    evidence_per_section = st.slider("Evidence/examples per section", 1, 6, 3)
with cols_top[3]:
    teacher_mode = st.toggle("Include Teacher View", value=True, help="Adds objectives, questions, activities, MCQs.")

show_devices_table = st.toggle("Show devices table (if applicable)", value=True)
show_line_by_line = st.toggle("Show line-by-line (for poetry/play)", value=True)

run = st.button("🔎 Analyze (Detailed)")

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
    st.info(f"Explanation language: **{explain_lang}** (detected from text: {detected})")

    # 1) Category detection (heuristic + GPT)
    guessed = heuristic_guess_category(source_text)
    cat = classify_with_gpt(source_text, explain_lang) or guessed
    st.success(f"📚 Detected category: **{cat}**")

    # 2) Build category-specific prompt + call GPT-4o
    safe_text = make_classroom_safe(source_text)
    system_msg = (
        "You are a veteran literature teacher for school students. "
        "Analyze the text in a CLASSROOM-SAFE way. Avoid explicit/graphic language. "
        "Stick to evidence from the text."
        + (" Respond in Hindi." if explain_lang.startswith("hi") else " Respond in English.")
    )
    user_msg = f"TEXT TO ANALYZE (verbatim):\n{safe_text}\n\n{build_schema_prompt(cat, explain_lang, detail_level, evidence_per_section, teacher_mode)}"

    with st.spinner("Calling GPT-4o for detailed structured analysis…"):
        ok, content = call_azure_chat(
            [{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=0.15 if detail_level >= 4 else 0.1,
            max_tokens=3000,
            force_json=True
        )

    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in tighter student-safe mode…")
        ok, content = call_azure_chat(
            [{"role":"system","content":"You are a cautious school literature teacher. Avoid explicit terms; use neutral wording."},
             {"role":"user","content":f"Return JSON for category '{cat}' with detail level {detail_level} and ~{evidence_per_section} evidences per section in language {explain_lang}. Text:\n{safe_text}"}],
            temperature=0.0, max_tokens=2600, force_json=True
        )

    if not ok:
        st.error(content)
        st.stop()

    data = robust_parse(content) or {}
    if not isinstance(data, dict) or not data:
        st.error("Model did not return valid JSON.")
        st.stop()

    # 3) Display
    st.markdown("### ✅ Analysis")
    st.caption(f"Category: **{cat}**")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Literal meaning**")
        st.write(data.get("literal_meaning","—"))
        st.markdown("**Figurative meaning / themes**")
        st.write(data.get("figurative_meaning","—"))
    with cols[1]:
        st.markdown("**Tone & Mood**")
        st.write(data.get("tone_mood","—"))
        st.markdown("**One-sentence takeaway**")
        st.write(data.get("one_sentence_takeaway","—"))

    # Category-specific sections (optional renderers)
    if cat == "poetry":
        if show_line_by_line and data.get("line_by_line"):
            st.markdown("### 📖 Line-by-line")
            for i, it in enumerate(data["line_by_line"], start=1):
                st.markdown(f"**Line {i}:** {it.get('line','')}")
                st.write(it.get("explanation",""))
                dev = it.get("device_notes","")
                if dev:
                    st.caption(f"Device notes: {dev}")
                st.divider()
        if show_devices_table and data.get("devices"):
            st.markdown("### 🎭 Devices")
            st.table([{"device":d.get("name",""),"evidence":d.get("evidence",""),"why":d.get("explanation","")} for d in data["devices"]])
        if data.get("imagery_map"):
            st.markdown("### 🌈 Imagery map")
            st.table(data["imagery_map"])
        if data.get("symbol_table"):
            st.markdown("### 🔶 Symbols")
            st.table(data["symbol_table"])
        if data.get("structure_overview"):
            st.markdown("### 🧩 Structure overview")
            st.json(data["structure_overview"], expanded=False)

    if cat in ("play","story"):
        if data.get("characters"):
            st.markdown("### 👥 Characters")
            st.table(data["characters"])
        if cat == "story" and data.get("plot_points"):
            st.markdown("### 🧭 Plot points")
            st.table(data["plot_points"])
        if cat == "play" and show_line_by_line and data.get("dialogue_beats"):
            st.markdown("### 💬 Dialogue beats")
            st.table(data["dialogue_beats"])
        if data.get("conflict"):
            st.markdown("### ⚔️ Conflict")
            st.write(data["conflict"])

    if cat == "essay":
        if data.get("thesis"):
            st.markdown("### 🎯 Thesis")
            st.write(data["thesis"])
        if data.get("key_points"):
            st.markdown("### 📌 Key points")
            st.table(data["key_points"])
        if data.get("rhetorical_devices"):
            st.markdown("### ✨ Rhetorical devices")
            st.table(data["rhetorical_devices"])

    # Shared extras
    if data.get("vocabulary_glossary"):
        st.markdown("### 📒 Glossary")
        st.table(data["vocabulary_glossary"])
    if data.get("themes"):
        st.markdown("### 🧠 Themes")
        st.write("\n".join(f"• {t}" for t in data["themes"]))
    if data.get("misconceptions"):
        st.markdown("### ⚠️ Misconceptions to avoid")
        st.write("\n".join(f"• {m}" for m in data["misconceptions"]))

    # Teacher view
    if data.get("teacher_view"):
        tv = data["teacher_view"]
        st.markdown("## 🧑‍🏫 Teacher View")
        if tv.get("learning_objectives"):
            st.markdown("**Learning Objectives**")
            st.write("\n".join(f"• {o}" for o in tv["learning_objectives"]))
        if tv.get("discussion_questions"):
            st.markdown("**Discussion Questions**")
            st.write("\n".join(f"• {q}" for q in tv["discussion_questions"]))
        if tv.get("classroom_activity"):
            st.markdown("**Classroom Activity**")
            for act in tv["classroom_activity"]:
                st.markdown(f"- **{act.get('title','Activity')}** ({act.get('duration_min','?')} min)")
                steps = act.get("steps") or []
                for i, sstep in enumerate(steps, start=1):
                    st.write(f"  {i}. {sstep}")
        if tv.get("quick_assessment_mcq"):
            st.markdown("**Quick Assessment (MCQ)**")
            st.table(tv["quick_assessment_mcq"])

    # 4) Raw JSON download
    with st.expander("🔧 Debug / Raw JSON"):
        st.json(data, expanded=False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_{cat}_analysis_{ts}.json",
            mime="application/json",
        )
