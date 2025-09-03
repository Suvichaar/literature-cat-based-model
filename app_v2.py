# app.py
import os
import re
import json
import html
from datetime import datetime
import requests
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar — Line-by-Line Explainer", page_icon="📚", layout="wide")
st.title("📚 Suvichaar — Line-by-Line Explainer (Azure)")
st.caption("Paste text or upload an image/PDF ➜ OCR (Azure DI) ➜ Auto-detect category ➜ Line/beat/paragraph explanations (Azure OpenAI) ➜ Export.")

# --- Azure Document Intelligence SDK (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

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
if (DocumentIntelligenceClient is None or AzureKeyCredential is None) and (AZURE_DI_ENDPOINT or AZURE_DI_KEY):
    st.info("Azure Document Intelligence SDK not installed. OCR will be skipped unless you install the SDK.")

# =========================
# HELPERS: OCR + SAFETY + AZURE CHAT
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

def make_classroom_safe(text: str) -> str:
    """Replace potentially sensitive words with classroom-friendly phrasing."""
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

def call_azure_chat(messages, *, temperature=0.1, max_tokens=5000, force_json=True):
    """Calls Azure OpenAI Chat Completions."""
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
# CATEGORY DETECTION (FAST HEURISTIC)
# =========================
CATEGORIES = [
    "poetry", "play", "story", "essay", "biography", "autobiography",
    "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"
]

def detect_hi_or_en(text: str) -> str:
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total = devanagari + latin
    if total == 0:
        return "en"
    return "hi" if (devanagari / total) >= 0.25 else "en"

def heuristic_guess_category(txt: str) -> str:
    t = txt.strip()
    lower = t.lower()
    if re.search(r'^\s*[A-Z][^\n]+:\s*', t, flags=re.MULTILINE):  # Speaker: line
        return "play"
    if re.search(r'^[^\n]{0,80}(\n[^\n]{0,80}){1,4}(\n|$)', t) and re.search(r'[,.!?]\s*$', t) is None:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) <= 40:
            return "poetry"
    if any(k in lower for k in ["dear sir", "yours faithfully", "yours sincerely", "subject:"]):
        return "letter"
    if any(k in lower for k in ["dear diary", "date:"]):
        return "diary"
    if any(k in lower for k in ["once upon a time", "moral"]):
        return "folk_tale"
    if any(k in lower for k in ["according to", "in conclusion", "therefore", "on the other hand"]):
        return "essay"
    if any(k in lower for k in ["i was born", "my childhood"]) and " i " in lower:
        return "autobiography"
    if any(k in lower for k in ["he was born", "she was born", "died in", "awarded"]):
        return "biography"
    if any(k in lower for k in ["ladies and gentlemen", "audience", "i stand before you"]):
        return "speech"
    if any(k in lower for k in ["report on", "findings", "method", "observation"]):
        return "report"
    return "story"  # default

def classify_with_gpt(txt: str, lang: str) -> str:
    """Ask Azure to classify; fall back to heuristic if anything fails."""
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
    label = re.sub(r'[^a-z_]', '', out.strip().lower())
    return label if label in CATEGORIES else heuristic_guess_category(txt)

# =========================
# SEGMENTERS (poetry / play / prose)
# =========================
def segment_poetry(text: str, max_segments: int):
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    segs = [{"label": f"Line {i+1}", "text": ln} for i, ln in enumerate(lines)]
    return segs[:max_segments] if max_segments else segs

def segment_play(text: str, max_segments: int):
    # Matches "SPEAKER: line...", tolerant of names with spaces/dots
    pattern = re.compile(r"^\s*([A-Z][A-Za-z .'-]{0,40}):\s*(.+)$")
    segs = []
    for i, ln in enumerate(text.splitlines()):
        m = pattern.match(ln)
        if m:
            speaker, speech = m.group(1).strip(), m.group(2).strip()
            segs.append({"label": f"{speaker}", "text": speech})
        else:
            if ln.strip():
                segs.append({"label": "Stage/Direction", "text": ln.strip()})
    return segs[:max_segments] if max_segments else segs

def split_paragraphs(text: str):
    # Merge wrapped lines into paragraphs; treat blank lines as separators
    chunks, buf = [], []
    for ln in text.splitlines():
        if ln.strip() == "":
            if buf:
                chunks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(ln.strip())
    if buf:
        chunks.append(" ".join(buf).strip())
    # Fallback to sentence-ish chunks if one giant blob
    if len(chunks) <= 1 and len(text) > 1000:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
        chunks = [p.strip() for p in parts if p.strip()]
    return chunks

def segment_prose(text: str, max_segments: int):
    paras = split_paragraphs(text)
    segs = [{"label": f"Paragraph {i+1}", "text": p} for i, p in enumerate(paras)]
    return segs[:max_segments] if max_segments else segs

# =========================
# PROMPT FOR EXPLANATIONS
# =========================
def build_explain_prompt(category: str, language_code: str, segments: list) -> list:
    """
    Returns Azure Chat messages asking for ONLY JSON:
    {
      "category": "...",
      "language": "en|hi",
      "segments": [
        {"label":"Line 1","text":"...","explanation":"...", "notes": "..."},
        ...
      ],
      "summary": "2-4 sentence overall summary"
    }
    """
    sys = (
        "You are an expert literature teacher for school students. "
        "Explain in a CLASSROOM-SAFE, neutral way with simple vocabulary."
        + (" Respond in Hindi." if language_code.startswith("hi") else " Respond in English.")
    )
    rules = (
        "Return ONLY JSON with this shape:\n"
        "{\n"
        '  "category": "poetry|play|story|essay|...",\n'
        '  "language": "en|hi",\n'
        '  "segments": [\n'
        '    {"label":"...","text":"...","explanation":"...", "notes":"(optional devices, tone, context)"}\n'
        "  ],\n"
        '  "summary": "2-4 sentences"\n'
        "}\n\n"
        "HARD RULES:\n"
        "- Keep the original text for each segment.\n"
        "- Provide clear, concise explanations (2–4 lines each).\n"
        "- For poetry: explain imagery/devices briefly; for plays: explain what the line/beat means; for prose: capture main idea of the paragraph.\n"
        "- No content outside JSON. No markdown fences.\n"
    )

    user = {
        "category": category,
        "language": language_code,
        "segments": segments
    }

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": rules + "\nINPUT:\n" + json.dumps(user, ensure_ascii=False)}
    ]

# =========================
# HTML EXPORT (simple portable page)
# =========================
def build_portable_html(bundle: dict) -> str:
    cat = bundle.get("category","").title()
    language = bundle.get("language","en")
    segments = bundle.get("segments",[]) or []
    summary = bundle.get("summary","") or ""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows = ""
    for seg in segments:
        label = html.escape(str(seg.get("label","")))
        text_ = html.escape(str(seg.get("text","")))
        expl  = html.escape(str(seg.get("explanation","")))
        notes = html.escape(str(seg.get("notes","")))
        rows += (
            "<tr>"
            f"<td style='white-space:nowrap'>{label}</td>"
            f"<td>{text_}</td>"
            f"<td>{expl}</td>"
            f"<td>{notes}</td>"
            "</tr>"
        )

    html_doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>'
        f"<title>Line-by-Line — {html.escape(cat)}</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;line-height:1.5;margin:24px;color:#0f172a}"
        "h1,h2{margin:0.4em 0}"
        ".card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;background:#fff}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}"
        ".small{color:#475569;font-size:12px}"
        "td:nth-child(2){width:36%} td:nth-child(3){width:36%}"
        "</style></head><body>"
        f"<h1>Line-by-Line — {html.escape(cat)} ({html.escape(language)})</h1>"
        f'<div class="small">Generated: {html.escape(ts)}</div>'

        f'<div class="card"><h2>Overall Summary</h2><p>{html.escape(summary or "—")}</p></div>'

        '<div class="card"><h2>Explanations</h2>'
        '<table><thead><tr><th>Label</th><th>Original</th><th>Explanation</th><th>Notes</th></tr></thead>'
        f"<tbody>{rows or '<tr><td colspan=\"4\">—</td></tr>'}</tbody></table></div>"

        '<div class="small">© Suvichaar</div>'
        "</body></html>"
    )
    return html_doc

# =========================
# UI CONTROLS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a poem / play / story / essay (optional)", height=180, placeholder="e.g., Whose woods these are I think I know…")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)

cols = st.columns(4)
with cols[0]:
    lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)
with cols[1]:
    max_segments = st.slider("Max lines/beats/paragraphs", 5, 150, 60, help="Cap to avoid token limits.")
with cols[2]:
    detail_level = st.slider("Detail level", 1, 5, 4, help="Higher → slightly more verbose.")
with cols[3]:
    category_override = st.selectbox("Force category (optional)", ["Auto", "poetry", "play", "story", "essay", "biography", "autobiography", "speech", "letter", "diary", "report", "folk_tale", "myth", "legend"], index=0)

run = st.button("🔎 Analyze & Explain")

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
    detected_lang = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else "hi" if lang_choice == "Hindi" else detected_lang

    # Category detection
    guessed = heuristic_guess_category(source_text)
    if category_override != "Auto":
        cat = category_override
    else:
        try:
            cat = classify_with_gpt(source_text, explain_lang) or guessed
        except Exception:
            cat = guessed

    st.markdown("#### Detected")
    st.markdown(f"- **Language:** `{explain_lang}`  \n- **Category:** `{cat}`")

    # Segment the text for explanation
    safe_text = make_classroom_safe(source_text)
    if cat == "poetry":
        segments = segment_poetry(safe_text, max_segments)
    elif cat == "play":
        segments = segment_play(safe_text, max_segments)
    else:
        segments = segment_prose(safe_text, max_segments)

    if not segments:
        st.error("No explainable segments detected.")
        st.stop()

    # Build prompt and call model (segment explanations)
    messages = build_explain_prompt(cat, explain_lang, segments)
    with st.spinner("Calling Azure to generate line-by-line explanations…"):
        ok, content = call_azure_chat(
            messages,
            temperature=0.25 if detail_level >= 4 else 0.1,
            max_tokens=6000,
            force_json=True
        )

    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in tighter student-safe mode…")
        messages_safe = [
            {"role":"system","content":"You are a cautious school literature teacher. Avoid explicit terms; use neutral wording."},
            messages[1]  # re-use user content
        ]
        ok, content = call_azure_chat(messages_safe, temperature=0.0, max_tokens=5500, force_json=True)

    if not ok:
        st.error(content)
        st.stop()

    parsed = robust_parse(content)
    out_category = parsed.get("category", cat)
    out_language = parsed.get("language", explain_lang)
    out_segments = parsed.get("segments", []) if isinstance(parsed, dict) else []
    out_summary = parsed.get("summary", "")

    # Show results
    st.markdown("### 📖 Explanations")
    if out_summary:
        st.info(out_summary)

    # Present as a table
    if out_segments:
        # render in chunks to avoid Streamlit length issues
        for idx in range(0, len(out_segments), 50):
            chunk = out_segments[idx:idx+50]
            rows = []
            for seg in chunk:
                rows.append({
                    "Label": seg.get("label",""),
                    "Original": seg.get("text",""),
                    "Explanation": seg.get("explanation",""),
                    "Notes": seg.get("notes","")
                })
            st.table(rows)

    # Export
    st.markdown("### ⬇️ Export")
    bundle = {
        "category": out_category,
        "language": out_language,
        "segments": out_segments,
        "summary": out_summary,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    html_str = build_portable_html(bundle)
    st.download_button(
        "Download portable HTML",
        data=html_str.encode("utf-8"),
        file_name=f"lit_explainer_{out_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True
    )
    st.download_button(
        "Download JSON",
        data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"lit_explainer_{out_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
