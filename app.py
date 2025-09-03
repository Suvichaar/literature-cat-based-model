# app.py
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st

# plotting
import numpy as np
import matplotlib.pyplot as plt

# Optional: only if you want S3 upload
try:
    import boto3
except Exception:
    boto3 = None

# Azure OpenAI SDK (v1+)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Math → Animated Steps (KaTeX) + Graph",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Math → Animated HTML (KaTeX) + 📈 Graph (if possible)")
st.caption("Type or upload a math problem → GPT returns concise LaTeX steps + optional graph data → we generate an animated .html file and plot a graph when appropriate.")


# =========================
# READ SECRETS / CONFIG
# =========================
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

AZURE_API_KEY      = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT     = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT   = get_secret("AZURE_DEPLOYMENT")          # e.g., "gpt-5-chat"
AZURE_API_VERSION  = get_secret("AZURE_API_VERSION", "2025-01-01-preview")

AWS_ACCESS_KEY     = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY     = get_secret("AWS_SECRET_KEY")
AWS_REGION         = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET         = get_secret("AWS_BUCKET")
S3_PREFIX          = get_secret("S3_PREFIX", "media")

CDN_HTML_BASE      = get_secret("CDN_HTML_BASE", "")  # e.g., https://stories.example.org/


# =========================
# AZURE CLIENT
# =========================
def get_azure_client():
    if AzureOpenAI is None:
        st.error("AzureOpenAI SDK not installed. Run:  pip install openai>=1.13.3")
        st.stop()
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        st.error("Azure OpenAI secrets missing. Add to .streamlit/secrets.toml.")
        st.stop()
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = """You are a helpful math/physics tutor.

Return a concise, correct solution as STRICT JSON ONLY (no prose outside JSON).
Use LaTeX for math (KaTeX-compatible). Keep steps short (1–2 lines each).

SCHEMA:
{
  "problem": "<original problem text>",
  "topic": "algebra|calculus|trig|mechanics|... (one word if possible)",
  "steps": [
    {"title": "Given", "latex": "..."},
    {"title": "Rule", "latex": "..."},
    {"title": "Apply", "latex": "..."}
  ],
  "final_answer_latex": "...",

  "graph": {
    "can_plot": true|false,
    "reason": "why graphable or not (1 sentence)",
    "kind": "function|parametric|points|implicit",
    "xlabel": "x (optional)",
    "ylabel": "y (optional)",
    "title": "short title (optional)",

    // IMPORTANT: to avoid code execution by the client,
    // you MUST include precomputed sample pairs to plot.
    // For function/implicit: N pairs [x, y].
    // For parametric: N pairs [x, y] in order of parameter.
    // For points: N pairs [x, y] as given.
    "samples": [[x1, y1], [x2, y2], ...]   // 20–300 points depending on smoothness
  }
}

RULES:
- Output must be valid JSON (no trailing commas).
- Escape backslashes correctly in LaTeX.
- If a graph is not meaningful, set can_plot=false and fill "reason".
- NEVER return code to be executed; only numeric samples for plotting.
- Prefer x range near the problem's natural domain.
"""

USER_PROMPT_TEMPLATE = """Problem:
{problem}

Constraints:
- Format JSON exactly as described. No extra keys.
- Use at most 6 steps.
- Prefer aligned equations where helpful:
  "\\begin{{aligned}} ... \\end{{aligned}}"
"""


# =========================
# GPT CALL
# =========================
def call_gpt_solve(problem_text: str) -> Dict[str, Any]:
    client = get_azure_client()

    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=problem_text.strip())},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content

    # Robust JSON parsing
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}\s*$", content)
        if not match:
            raise ValueError("Model did not return JSON.")
        data = json.loads(match.group(0))
    return data


# =========================
# HTML GENERATOR (KaTeX + Animation)
# =========================
def build_animated_katex_html(payload: Dict[str, Any]) -> str:
    safe_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Solution — Animated Steps</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{{ --bg:#0d1117; --card:#141b23; --text:#eaf2f8; --muted:#9fb6c2; --stroke:#223140; --accent:#7ee787; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,Ubuntu,sans-serif}}
header{{padding:16px 20px;border-bottom:1px solid var(--stroke)}}
h1{{margin:0 0 6px;font-size:20px}}
h2{{margin:10px 0 8px}}
small, .muted{{color:var(--muted)}}
.wrap{{max-width:1100px;margin:18px auto 48px;padding:0 16px}}
.panel{{background:var(--card);border:1px solid var(--stroke);border-radius:14px;padding:16px}}
.controls .btn{{padding:10px 14px;border-radius:10px;border:1px solid var(--stroke);background:#101824;color:var(--text);cursor:pointer;font-weight:600}}
.controls .btn:active{{transform:translateY(1px)}}
.controls .btn.accent{{outline:2px solid var(--accent)}}
.steps{{list-style:none;padding:0;margin:16px 0 0}}
.step{{background:#0f1620;border:1px solid var(--stroke);border-radius:12px;padding:14px;margin:10px 0;overflow:hidden;opacity:0;transform:translateY(10px);transition:opacity .35s ease, transform .35s ease}}
.step.show{{opacity:1;transform:none}}
.math{{min-height:36px;max-width:100%;overflow-x:auto}}
.katex-display{{margin:.2rem 0}}
.katex-display>.katex{{white-space:normal}}
.progress{{height:6px;border-radius:999px;background:#0e1520;border:1px solid var(--stroke);overflow:hidden;margin:10px 0 0}}
.progress>div{{height:100%;width:0%;background:var(--accent);transition:width .3s ease}}
.answer{{background:#0c151f;border:1px dashed var(--stroke);border-radius:12px;padding:12px;margin-top:14px}}
</style>
</head>
<body>
<header>
  <h1>Solution — Animated Steps</h1>
  <div class="muted" id="meta"></div>
</header>

<div class="wrap">
  <div class="panel">
    <div class="controls">
      <button id="play" class="btn accent">▶ Play</button>
      <button id="step" class="btn">→ Step</button>
      <button id="pause" class="btn">⏸ Pause</button>
      <button id="reset" class="btn">↺ Reset</button>
    </div>
    <div class="progress"><div id="bar"></div></div>
    <ol id="steps" class="steps"></ol>
    <div class="answer" id="answer"></div>
  </div>
</div>

<script>
const payload = {safe_json};

function katexRender(el, tex, display=true) {{
  try {{ katex.render(tex, el, {{throwOnError:false, displayMode:display}}); }}
  catch(e) {{ el.textContent = tex; }}
}}

const list = document.getElementById('steps');
const bar  = document.getElementById('bar');
const ans  = document.getElementById('answer');
const meta = document.getElementById('meta');

meta.textContent = "Topic: " + (payload.topic||"") + "   •   Problem: " + (payload.problem||"");

let i=-1, playing=false;

function build() {{
  list.innerHTML = "";
  ans.innerHTML = "";
  (payload.steps||[]).forEach((s, idx) => {{
    const li = document.createElement('li');
    li.className = 'step';
    li.innerHTML = `<h2>${{s.title||('Step '+(idx+1))}}</h2><div class="math"></div>`;
    list.appendChild(li);
    s._node = li.querySelector('.math');
  }});
  i=-1; updateBar();
  ans.innerHTML = "<small>Final answer</small><div id='ansmath'></div>";
}}

function updateBar() {{
  const pct = Math.max(0, (i+1)/((payload.steps||[]).length)) * 100;
  bar.style.width = pct + '%';
}}

function next() {{
  if (!payload.steps || i >= payload.steps.length-1) return;
  i++;
  const s = payload.steps[i];
  s._node.parentElement.classList.add('show');
  katexRender(s._node, s.latex || "", true);
  updateBar();
  if (i === payload.steps.length-1) {{
    const el = document.getElementById('ansmath');
    katexRender(el, payload.final_answer_latex || "", true);
  }}
}}

document.getElementById('play').onclick  = () => {{
  if (playing) return;
  playing = true;
  (function loop() {{
    if (!playing) return;
    if (i < (payload.steps||[]).length-1) {{
      next(); setTimeout(loop, 700);
    }} else {{
      playing = false;
    }}
  }})();
}};
document.getElementById('step').onclick  = () => {{ playing=false; next(); }};
document.getElementById('pause').onclick = () => {{ playing=false; }};
document.getElementById('reset').onclick = () => {{ playing=false; build(); }};

build();
</script>
</body>
</html>
"""
    return html


# =========================
# UTIL
# =========================
def save_html(html: str, filename: str) -> Path:
    out = Path("animated_exports")
    out.mkdir(exist_ok=True)
    p = out / filename
    p.write_text(html, encoding="utf-8")
    return p


def s3_upload(file_path: Path, key: str) -> Optional[str]:
    if boto3 is None:
        st.error("boto3 not installed. Run: pip install boto3")
        return None
    if not (AWS_BUCKET and (AWS_ACCESS_KEY and AWS_SECRET_KEY or True)):
        st.error("Missing S3 config in secrets.")
        return None

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    s3 = session.client("s3")
    s3.upload_file(str(file_path), AWS_BUCKET, key, ExtraArgs={"ContentType": "text/html", "ACL": "public-read"})
    if CDN_HTML_BASE:
        return f"{CDN_HTML_BASE}{key}"
    return f"s3://{AWS_BUCKET}/{key}"


def make_slug(s: str, n: int = 36) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip()).strip("-").lower()
    return (s[:n] or "solution")


# =========================
# GRAPH RENDERING
# =========================
def render_graph(payload: Dict[str, Any]) -> None:
    """
    Draw a graph if payload['graph'] says it's possible.
    Expect structure:
      graph = {
        "can_plot": bool, "reason": str, "kind": "function|parametric|points|implicit",
        "xlabel": "...", "ylabel": "...", "title": "...",
        "samples": [[x,y], ...]
      }
    """
    graph = payload.get("graph") or {}
    can_plot = bool(graph.get("can_plot"))
    reason = graph.get("reason", "")

    st.subheader("📈 Graph")
    if not can_plot:
        msg = reason or "Graph not available for this question."
        st.warning(f"Graph not available: {msg}")
        return

    samples = graph.get("samples") or []
    if not samples:
        st.warning("Graph not available: no sample data provided.")
        return

    # Convert to arrays
    try:
        arr = np.array(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Invalid samples shape")
    except Exception:
        st.warning("Graph not available: invalid sample data.")
        return

    x, y = arr[:, 0], arr[:, 1]

    fig, ax = plt.subplots()            # 1 chart only, no explicit colors
    ax.plot(x, y)
    ax.set_xlabel(graph.get("xlabel") or "x")
    ax.set_ylabel(graph.get("ylabel") or "y")
    ax.set_title(graph.get("title") or "Graph")
    ax.grid(True)

    st.pyplot(fig)


# =========================
# SIDEBAR / INPUTS
# =========================
with st.sidebar:
    st.header("Input")
    default_example = "A thin ring (mass M, radius r) rotates with ω. Two masses m are attached diametrically. Find new angular velocity."
    src = st.radio("How to supply a problem?", ["Type it", "Upload .txt"], index=0)
    problem_text = ""
    if src == "Type it":
        problem_text = st.text_area("Enter a math/physics problem", value=default_example, height=140)
    else:
        up = st.file_uploader("Upload a .txt file", type=["txt"])
        if up:
            problem_text = up.read().decode("utf-8")

    st.markdown("---")
    autoupload = st.checkbox("Upload result to S3 after generation", value=False)
    prefix = st.text_input("S3 prefix (folder)", value=S3_PREFIX or "media")
    st.caption("File is public-read. Ensure bucket CORS/ACL/policy allow it.")


# =========================
# MAIN ACTION
# =========================
left, right = st.columns([0.38, 0.62])

with left:
    if st.button("🚀 Solve & Generate Animated HTML", use_container_width=True):
        if not problem_text.strip():
            st.error("Please provide a problem.")
            st.stop()

        with st.spinner("Asking GPT for LaTeX steps + graphability..."):
            try:
                result = call_gpt_solve(problem_text)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.success("Got result from GPT!")
        st.json(result)

        with st.spinner("Building animated KaTeX HTML..."):
            html = build_animated_katex_html(result)
            stem = make_slug(result.get("topic") or "math") + "-" + str(int(time.time()))
            filename = f"{stem}.html"
            path = save_html(html, filename)

        st.success(f"HTML written: {path}")

        # Preview
        with open(path, "r", encoding="utf-8") as f:
            html_str = f.read()
        st.components.v1.html(html_str, height=560, scrolling=True)

        # Download button
        st.download_button(
            "⬇️ Download HTML",
            data=html_str,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )

        # S3 upload (optional)
        if autoupload:
            key = f"{(prefix or 'media').strip('/')}/{filename}"
            with st.spinner(f"Uploading to s3://{AWS_BUCKET}/{key}"):
                try:
                    url = s3_upload(path, key)
                except Exception as e:
                    st.exception(e)
                    url = None
            if url:
                st.success("Uploaded!")
                st.write("Public URL:")
                st.code(url)

        # store for right column
        st.session_state["last_payload"] = result

with right:
    if "last_payload" in st.session_state:
        render_graph(st.session_state["last_payload"])
    else:
        st.info("Run a problem to see the graph area. If the problem is not graphable, you’ll see a friendly notice.")
