# app.py
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Azure OpenAI SDK
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Math → Animated Steps (KaTeX) + Matplotlib PNG",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Math → Animated HTML (KaTeX) + 🖼️ Matplotlib PNG")
st.caption("Type or upload a problem → GPT returns concise LaTeX steps → we build KaTeX HTML + save solution frames as .png")

# =========================
# READ SECRETS
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
        st.error("AzureOpenAI SDK missing. Install: pip install openai>=1.13.3")
        st.stop()
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        st.error("Missing Azure secrets in .streamlit/secrets.toml")
        st.stop()
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

# =========================
# GPT PROMPTS
# =========================
SYSTEM_PROMPT = """You are a helpful math/physics tutor.
Return STRICT JSON ONLY (no prose).
Use KaTeX-compatible LaTeX. Use ≤6 steps.
For Matplotlib rendering, prefer mathtext-safe LaTeX (avoid \text{}, aligned, eqnarray; prefer \mathrm{})."""

USER_PROMPT_TEMPLATE = """Problem:
{problem}
Constraints:
- JSON only, ≤6 steps
- Prefer mathtext-safe LaTeX
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
            raise ValueError("Model returned invalid JSON")
        return json.loads(match.group(0))

# =========================
# HTML BUILDER
# =========================
def build_animated_katex_html(payload: Dict[str, Any]) -> str:
    safe_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Solution Steps</title>
<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'>
<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>
</head>
<body>
<h1>Solution — Animated Steps</h1>
<div id='meta'></div>
<ol id='steps'></ol>
<div id='answer'></div>
<script>
const payload = {safe_json};
const stepsEl = document.getElementById('steps');
const ansEl = document.getElementById('answer');
const meta = document.getElementById('meta');
meta.textContent = "Topic: " + (payload.topic||"") + " | Problem: " + (payload.problem||"");
(payload.steps||[]).forEach((s,i)=>{{
  const li = document.createElement('li');
  li.innerHTML = `<b>${{s.title||('Step '+(i+1))}}</b>: <span id='m${i}'></span>`;
  stepsEl.appendChild(li);
  katex.render(s.latex||'', document.getElementById(`m${i}`), {{throwOnError:false}});
}});
const ans = document.createElement('div');
ans.innerHTML = `<b>Final Answer:</b> <span id='ans'></span>`;
ansEl.appendChild(ans);
katex.render(payload.final_answer_latex||'', document.getElementById('ans'), {{throwOnError:false}});
</script>
</body>
</html>"""

# =========================
# MATPLOTLIB PNG EXPORT
# =========================
def save_solution_as_png(payload: Dict[str, Any], outfile: Path):
    import matplotlib.pyplot as plt
    import re as _re

    def sanitize(tex: str) -> str:
        if not isinstance(tex, str):
            return ""
        s = tex.strip()
        s = _re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", s)
        s = _re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", s)
        s = s.replace(r"\\", "\n")
        s = s.replace(r"\implies", r"\Rightarrow").replace(r"\iff", r"\Leftrightarrow")
        return s

    steps = payload.get("steps", [])
    final = payload.get("final_answer_latex", "")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.axis('off')
    y = 0.95
    ax.text(0.05, y, "Solution Steps", fontsize=14, weight='bold', transform=ax.transAxes)
    y -= 0.05
    for i, step in enumerate(steps):
        ax.text(0.05, y, f"{step.get('title','Step '+str(i+1))}:", fontsize=11, weight='bold', transform=ax.transAxes)
        y -= 0.05
        try:
            ax.text(0.08, y, f"$ {sanitize(step.get('latex',''))} $", fontsize=11, transform=ax.transAxes)
        except:
            ax.text(0.08, y, sanitize(step.get('latex','')), fontsize=11, transform=ax.transAxes)
        y -= 0.08
    if final:
        y -= 0.03
        ax.text(0.05, y, "Final Answer:", fontsize=12, weight='bold', transform=ax.transAxes)
        y -= 0.05
        try:
            ax.text(0.08, y, f"$ {sanitize(final)} $", fontsize=12, transform=ax.transAxes)
        except:
            ax.text(0.08, y, sanitize(final), fontsize=12, transform=ax.transAxes)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile

# =========================
# UTILS
# =========================
def save_html(html: str, filename: str) -> Path:
    out = Path("animated_exports")
    out.mkdir(exist_ok=True)
    p = out / filename
    p.write_text(html, encoding="utf-8")
    return p

def make_slug(s: str, n: int = 36) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s.strip()).strip("-").lower()[:n]

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
# MAIN
# =========================
if st.button("🚀 Solve & Generate (HTML + PNG)", use_container_width=True):
    if not problem_text.strip():
        st.error("Please provide a problem.")
        st.stop()

    with st.spinner("Asking GPT for LaTeX steps..."):
        try:
            result = call_gpt_solve(problem_text)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("Got result from GPT!")

    with st.spinner("Building HTML..."):
        html = build_animated_katex_html(result)
        stem = make_slug(result.get("topic") or "math") + "-" + str(int(time.time()))
        html_filename = f"{stem}.html"
        html_path = save_html(html, html_filename)

    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
    st.components.v1.html(html_str, height=640, scrolling=True)
    st.download_button("⬇️ Download HTML", data=html_str, file_name=html_filename, mime="text/html")

    with st.spinner("Saving steps as PNG..."):
        out_dir = Path("animated_exports")
        out_dir.mkdir(exist_ok=True)
        png_path = out_dir / (stem + ".png")
        save_solution_as_png(result, png_path)
        st.image(str(png_path), caption="Solution Steps", use_container_width=True)
        st.download_button("⬇️ Download PNG", data=png_path.read_bytes(), file_name=png_path.name, mime="image/png")

    with st.expander("Show raw JSON from GPT"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
