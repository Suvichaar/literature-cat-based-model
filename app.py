import os
import re
import json
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# Optional: OCR SDK (Azure Document Intelligence)
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar — Literature Insight (Template + Q&A)", page_icon="📚", layout="wide")
st.title("📚 Suvichaar — Literature Insight")
st.caption(
    "Upload/paste text → OCR → Auto-detect category → Editable category template → "
    "Generate structured analysis + short/long answers + quizzes."
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
        # new preview flag for strict JSON
        body["response_format"] = {"type": "json_object"}
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
# CATEGORY-SPECIFIC TEMPLATES (EDITABLE)
# =========================
TEMPLATES = {
    "poetry": {
        "sections": [
            "Title & Poet",
            "Speaker/Voice & Point of View",
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
            "Vocabulary & Meanings (archaic/Elizabethan etc.)",
            "Comparative Texts/Productions (optional)",
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
            "Tone & Mood / Style",
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
# PROMPT BUILDERS
# =========================
def build_analysis_prompt(category: str, language_code: str, detail: int, evidence_cap: int, selected_sections: list) -> str:
    """
    Ask the model to fill a compact 'data' object aligned to the chosen sections.
    """
    schema_hint = {
        "executive_summary": "4–6 lines overview",
        "inspiration_hook": "curiosity hook",
        "why_it_matters": "transferable skills/values",
        "study_tips": ["concise tips"],
        "extension_reading": ["related texts"]
    }
    return (
        "You are an expert literature teacher. Analyze the text below in a CLASSROOM-SAFE way.\n"
        "Return ONLY JSON with a single key 'data'. Keep arrays concise and capped.\n"
        f"Target language: {language_code}\n"
        f"Category: {category}\n"
        f"Detail level (1-5): {detail}\n"
        f"Max items per list: {evidence_cap}\n"
        "Required keys (as supported by text): executive_summary, inspiration_hook, why_it_matters, "
        "study_tips, extension_reading, themes_detailed, quote_bank.\n"
        "Also align content to these section headings from the teacher template:\n"
        + json.dumps(selected_sections, ensure_ascii=False)
        + "\nSchema hints:\n" + json.dumps(schema_hint, ensure_ascii=False, indent=2)
    )


def build_qa_prompt(kind: str, count: int, language_code: str, difficulty: str = "balanced") -> str:
    """
    kind: 'short', 'long', or 'quiz'
    - short → 1–2 line answers
    - long → 5–8 line model answers with quotes
    - quiz → MCQs with 4 options and correct answer key
    """
    if kind == "short":
        return (
            "Create short-answer Q&A for the given text. "
            f"Return ONLY JSON with key 'questions_short' as an array of {count} objects {{q,a}}.\n"
            "Answers should be 1-2 lines, classroom-safe, and evidence-based."
            f"\nLanguage: {language_code}. Difficulty: {difficulty}."
        )
    if kind == "long":
        return (
            "Create analytical long-answer Q&A for the given text. "
            f"Return ONLY JSON with key 'questions_long' as an array of {count} objects {{q,a}}.\n"
            "Each answer should be 5–8 lines and include brief textual evidence in quotes."
            f"\nLanguage: {language_code}. Difficulty: {difficulty}."
        )
    # quiz
    return (
        "Create multiple-choice questions (MCQs) for the given text. "
        f"Return ONLY JSON with key 'quiz' as an array of {count} objects "
        "{{q, choices:[A,B,C,D], answer}}.\n"
        "Questions must be unambiguous; one correct answer only; choices plausible."
        f"\nLanguage: {language_code}. Difficulty: {difficulty}."
    )


# =========================
# UI — INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste text (optional)", height=160, placeholder="Paste poem/play/story/essay here")
files = st.file_uploader("Or upload an image/PDF", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols = st.columns(5)
with cols[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols[1]:
    detail_level = st.slider("Detail level", 1, 5, 4)
with cols[2]:
    evidence_cap = st.slider("Max items per list", 1, 6, 3)
with cols[3]:
    gen_short = st.checkbox("Gen Short Answers")
with cols[4]:
    gen_long = st.checkbox("Gen Long Answers")

cols2 = st.columns(3)
with cols2[0]:
    gen_quiz = st.checkbox("Gen Quiz (MCQ)")
with cols2[1]:
    quiz_count = st.number_input("Quiz count", min_value=4, max_value=20, value=10, step=1)
with cols2[2]:
    difficulty = st.selectbox("Difficulty", ["easy", "balanced", "challenging"], index=1)

run = st.button("🔎 Analyze & Open Template")


# =========================
# MAIN
# =========================
if "template_df" not in st.session_state:
    st.session_state.template_df = None
if "category" not in st.session_state:
    st.session_state.category = None
if "source_text" not in st.session_state:
    st.session_state.source_text = ""

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

    # feedback chips
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

    # 1) Structured Analysis
    if regen_analysis:
        with st.spinner("Generating structured analysis…"):
            prompt = build_analysis_prompt(category, explain_lang, detail_level, evidence_cap, selected_sections)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.15 if detail_level >= 4 else 0.1,
                max_tokens=3500,
                force_json=True
            )
        if not ok and content == "FILTERED":
            st.warning("Sensitive content detected. Retrying with tighter safe mode…")
            ok, content = call_azure_chat(
                [{"role": "system", "content": "You are a cautious school literature teacher. Avoid explicit terms; use neutral wording."},
                 {"role": "user", "content": f"Return ONLY JSON with 'data' for category '{category}'. Keep arrays ≤ {evidence_cap}. Language {explain_lang}.\nTEXT:\n{safe_text}"}],
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
                st.markdown("#### 📦 Structured Analysis (JSON)")
                st.json(data, expanded=False)

    # 2) Short Answers
    if gen_short and make_short:
        with st.spinner("Generating short answers…"):
            prompt = build_qa_prompt("short", evidence_cap if evidence_cap >= 3 else 3, explain_lang)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.12, max_tokens=2000, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("questions_short")
            if not isinstance(arr, list):
                st.error("Expected key 'questions_short' as an array.")
            else:
                st.markdown("#### ❔ Short Q&A")
                st.table(arr)

    # 3) Long Answers
    if gen_long and make_long:
        with st.spinner("Generating long answers…"):
            prompt = build_qa_prompt("long", max(3, evidence_cap), explain_lang, difficulty=difficulty)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.15, max_tokens=2800, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("questions_long")
            if not isinstance(arr, list):
                st.error("Expected key 'questions_long' as an array.")
            else:
                st.markdown("#### 🧩 Long Q&A (Analytical)")
                st.table(arr)

    # 4) Quiz
    if gen_quiz and make_quiz:
        with st.spinner("Generating quiz…"):
            prompt = build_qa_prompt("quiz", int(quiz_count), explain_lang, difficulty=difficulty)
            ok, content = call_azure_chat(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": f"TEXT:\n{safe_text}\n\n{prompt}"}],
                temperature=0.15, max_tokens=2500, force_json=True
            )
        if not ok:
            st.error(content)
        else:
            parsed = robust_parse(content) or {}
            arr = parsed.get("quiz")
            if not isinstance(arr, list):
                st.error("Expected key 'quiz' as an array.")
            else:
                st.markdown("#### 📝 Quiz (MCQ)")
                st.table(arr)
