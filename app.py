import json
import re
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st

# Azure OpenAI SDK (v1+)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Math → Animated Steps (KaTeX + TikZ + GIF)",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Math → Animated HTML (KaTeX) + TikZ + 🎞 GIF")
st.caption(
    "Type a math/physics problem → GPT returns LaTeX steps + TikZ → "
    "We generate animated KaTeX HTML + TikZ diagram + GIF fallback."
)


# =========================
# READ SECRETS / CONFIG
# =========================
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


AZURE_API_KEY = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = get_secret("AZURE_DEPLOYMENT")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2025-01-01-preview")


# =========================
# AZURE CLIENT
# =========================
def get_azure_client():
    if AzureOpenAI is None:
        st.error("AzureOpenAI SDK not installed. Run: pip install openai>=1.13.3")
        st.stop()
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        st.error("Azure OpenAI secrets missing in .streamlit/secrets.toml")
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

Return a concise, correct solution as STRICT JSON ONLY.
Use LaTeX for math. Keep steps short.

SCHEMA:
{
  "problem": "<original problem>",
  "topic": "algebra|calculus|trig|mechanics|...",
  "steps": [
    {"title": "Given", "latex": "..."},
    {"title": "Rule", "latex": "..."},
    {"title": "Apply", "latex": "..."}
  ],
  "final_answer_latex": "...",
  "tikz": {
    "can_draw": true|false,
    "reason": "why a diagram helps",
    "source": "\\n\\begin{tikzpicture}[scale=1]\\n ... \\n\\end{tikzpicture}\\n"
  }
}

RULES:
- Strict JSON only.
- Escape backslashes properly.
- Keep TikZ minimal and compile-ready.
"""

USER_PROMPT_TEMPLATE = """Problem:
{problem}

Constraints:
- Follow the JSON schema exactly.
- Max 6 steps.
- Use minimal TikZ if helpful (axes, labeled points, or one curve).
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
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}\s*$", content)
        if not match:
            raise ValueError("Model did not return JSON.")
        return json.loads(match.group(0))


# =========================
# GIF GENERATOR (TikZ → Approximation)
# =========================
def generate_gif_from_tikz(tikz_src: str, out_path: Path):
    """
    Approximate TikZ function plots by detecting y = f(x).
    If unsupported TikZ, returns None.
    """
    match = re.search(r"y\s*=\s*([x0-9\+\-\*\/\^\(\)\s]+)", tikz_src)
    if not match:
        return None
    expr = match.group(1)

    # Safe evaluation
    def safe_eval(x):
        try:
            return eval(expr.replace("^", "**"), {"x": x, "np": np})
        except Exception:
            return None

    x = np.linspace(-3, 3, 200)
    y = safe_eval(x)
    if y is None:
        return None

    fig, ax = plt.subplots()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Plot: y = {expr}")

    ax.axhline(0, color="gray")
    ax.axvline(0, color="gray")

    (line,) = ax.plot([], [], lw=2)
    frames = range(1, len(x) + 1, 4)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        line.set_data(x[:i], y[:i])
        return (line,)

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)
    anim.save(out_path, writer=PillowWriter(fps=24))
    plt.close(fig)
    return out_path


# =========================
# HTML GENERATOR (KaTeX + TikZ)
# =========================
def build_animated_katex_html(payload: Dict[str, Any]) -> str:
    safe_json = json.dumps(payload, ensure_ascii=False)
    tikz = (payload.get("tikz") or {})
    tikz_can = bool(tikz.get("can_draw"))
    tikz_src = tikz.get("source") or ""

    tikz_block = ""
    if tikz_can and tikz_src.strip():
        tikz_block = f"""
<section class="panel">
  <h2>TikZ Diagram</h2>
  <div class="muted">{(tikz.get("reason") or "").strip()}</div>
  <div class="tikz-wrap">
    <script type="text/tikz">
{tikz_src}
    </script>
  </div>
</section>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Solution — Animated Steps</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://tikzjax.com/v1/tikzjax.js"></script>
</head>
<body>
<h1>Solution Steps</h1>
<div id="meta"></div>
<ol id="steps"></ol>
<div id="answer"></div>
{tikz_block}
<script>
const payload = {safe_json};
</script>
</body>
</html>"""
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


def make_slug(s: str, n: int = 36) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip()).strip("-").lower()
    return (s[:n] or "solution")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Input")
    default_example = "Sketch y = x^2 - 1 and find its vertex and y-intercept."
    src = st.radio("How to supply a problem?", ["Type it", "Upload .txt"], index=0)
    problem_text = ""
    if src == "Type it":
        problem_text = st.text_area("Enter a math/physics problem", value=default_example, height=140)
    else:
        up = st.file_uploader("Upload a .txt file", type=["txt"])
        if up:
            problem_text = up.read().decode("utf-8")


# =========================
# MAIN ACTION
# =========================
if st.button("🚀 Solve & Generate Animated HTML", use_container_width=True):
    if not problem_text.strip():
        st.error("Please provide a problem.")
        st.stop()

    with st.spinner("Asking GPT for LaTeX + TikZ..."):
        try:
            result = call_gpt_solve(problem_text)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("Got result from GPT!")

    with st.spinner("Building HTML..."):
        html = build_animated_katex_html(result)
        stem = make_slug(result.get("topic") or "math") + "-" + str(int(time.time()))
        filename = f"{stem}.html"
        path = save_html(html, filename)

    with open(path, "r", encoding="utf-8") as f:
        html_str = f.read()

    # ---- Show GIF if TikZ is present ----
    tikz = result.get("tikz", {})
    if tikz.get("can_draw") and tikz.get("source"):
        st.subheader("🎞 TikZ Approximation GIF")
        gif_path = Path("animated_exports") / f"{stem}.gif"
        gif = generate_gif_from_tikz(tikz["source"], gif_path)
        if gif:
            st.image(str(gif), caption="TikZ-based Animated GIF")
            st.download_button(
                "⬇️ Download GIF",
                data=open(gif, "rb").read(),
                file_name=gif_path.name,
                mime="image/gif",
            )
        else:
            st.warning("TikZ diagram detected, but GIF approximation failed.")

    # ---- Open TikZ in New Tab ----
    if '<script type="text/tikz">' in html_str:
        st.info("TikZ may not render in Streamlit preview → Open in a new tab:")
        b64 = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
        data_url = f"data:text/html;base64,{b64}"
        st.markdown(f"[🔎 Open TikZ Preview in New Tab]({data_url})", unsafe_allow_html=True)

        # Show TikZ source code
        m = re.search(r'<script type="text/tikz">\s*([\s\S]*?)\s*</script>', html_str)
        if m:
            st.caption("TikZ source:")
            st.code(m.group(1).strip(), language="latex")

    # HTML Preview (steps only, TikZ may not render)
    st.components.v1.html(html_str, height=640, scrolling=True)

    # Download HTML
    st.download_button(
        "⬇️ Download HTML",
        data=html_str,
        file_name=filename,
        mime="text/html",
    )
