# app.py
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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
    page_title="Math → Animated Steps (KaTeX) + Optional Graph",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Math → Animated HTML (KaTeX) + 🖼️ Optional Matplotlib Graph")
st.caption(
    "Type or upload a problem → GPT returns concise LaTeX steps → we build KaTeX HTML. "
    "If you tick the checkbox, we also parse and plot y = f(x) as a PNG."
)

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
For Matplotlib rendering, prefer mathtext-safe LaTeX (avoid \\text{}, aligned, eqnarray; prefer \\mathrm{})."""

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
# HTML BUILDER (KaTeX Animated)
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
<style>
:root{{ --bg:#0d1117; --card:#141b23; --text:#eaf2f8; --muted:#9fb6c2; --stroke:#223140; --accent:#7ee787; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,Ubuntu,sans-serif}}
header{{padding:16px 20px;border-bottom:1px solid var(--stroke)}}
h1{{margin:0 0 6px;font-size:20px}}
h2{{margin:10px 0 8px}}
small, .muted{{color:var(--muted)}}
.wrap{{max-width:980px;margin:18px auto 48px;padding:0 16px}}
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
  <div class='muted' id='meta'></div>
</header>
<div class='wrap'>
  <section class='panel'>
    <div class='controls'>
      <button id='play' class='btn accent'>▶ Play</button>
      <button id='step' class='btn'>→ Step</button>
      <button id='pause' class='btn'>⏸ Pause</button>
      <button id='reset' class='btn'>↺ Reset</button>
    </div>
    <div class='progress'><div id='bar'></div></div>
    <ol id='steps' class='steps'></ol>
    <div class='answer' id='answer'></div>
  </section>
</div>
<script>
const payload = {safe_json};
function katexRender(el, tex, display=true){
  try{ katex.render(tex, el, {throwOnError:false, displayMode:display}); }catch(e){ el.textContent = tex; }
}
const list=document.getElementById('steps');
const bar=document.getElementById('bar');
const ans=document.getElementById('answer');
const meta=document.getElementById('meta');
meta.textContent = "Topic: "+(payload.topic||"")+"   •   Problem: "+(payload.problem||"");
let i=-1, playing=false;
function build(){
  list.innerHTML=""; ans.innerHTML="";
  (payload.steps||[]).forEach((s,idx)=>{
    const li=document.createElement('li');
    li.className='step';
    li.innerHTML=`<h2>${s.title||('Step '+(idx+1))}</h2><div class='math'></div>`;
    list.appendChild(li);
    s._node=li.querySelector('.math');
  });
  i=-1; updateBar();
  ans.innerHTML = "<small>Final answer</small><div id='ansmath'></div>";
}
function updateBar(){ bar.style.width = Math.max(0,(i+1)/((payload.steps||[]).length))*100+'%'; }
function next(){
  if(!payload.steps||i>=(payload.steps.length-1)) return;
  i++; const s=payload.steps[i]; s._node.parentElement.classList.add('show');
  katexRender(s._node, s.latex||"", true); updateBar();
  if(i===(payload.steps.length-1)) katexRender(document.getElementById('ansmath'), payload.final_answer_latex||"", true);
}
function playLoop(){ if(!playing) return; if(i<(payload.steps||[]).length-1){ next(); setTimeout(playLoop,700);} else {playing=false;} }
play.onclick=()=>{ if(!playing){ playing=true; playLoop(); } };
step.onclick=()=>{ playing=false; next(); };
pause.onclick=()=>{ playing=false; } };
reset.onclick=()=>{ playing=false; build(); };
build();
</script>
</body>
</html>"""

# =========================
# GRAPH PARSING & PLOTTING (Optional PNG)
# =========================

def guess_equation(text: str) -> Optional[str]:
    """Best-effort grab of 'y = ...' (or f(x)=...) from the prompt text."""
    if not text:
        return None
    m = re.search(r"y\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
    if m:
        rhs = m.group(1).strip()
        return f"y = {rhs}"
    m = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
    if m:
        rhs = m.group(1).strip()
        return f"y = {rhs}"
    # Last resort: a bare expression containing 'x' (e.g., x^2 - 1)
    m = re.search(r"(?<![A-Za-z0-9_])(x[^\n;]+)", text)
    if m:
        cand = m.group(1).strip()
        return f"y = {cand}"
    return None


def _to_numpy_expr(expr: str) -> str:
    """Convert a math string into a numpy-evaluable Python expression."""
    s = expr.strip()
    # take RHS if 'y = ...'
    if re.match(r"^y\s*=", s, flags=re.IGNORECASE):
        s = s.split("=", 1)[1]
    # basic normalizations
    s = s.replace("^", "**")
    s = s.replace("ln", "log")
    # implicit multiplication: 2x -> 2*x, 2(x+1)->2*(x+1)
    s = re.sub(r"(?<=\d)\s*(?=x)", "*", s)
    s = re.sub(r"(?<=\d)\s*\(", "*(", s)
    # Allow absolute value via abs()
    s = s.replace("|x|", "abs(x)")
    return s


def plot_equation_png(equation: str, outfile: Path, x_min: float = -5.0, x_max: float = 5.0, points: int = 1000) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    expr = _to_numpy_expr(equation)

    # Safe eval environment
    x = np.linspace(x_min, x_max, int(points))
    env = {
        "x": x,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }

    try:
        y = eval(expr, {"__builtins__": {}}, env)
    except Exception as e:
        raise ValueError(f"Could not parse/evaluate the equation: {equation}\nDetails: {e}")

    if hasattr(y, "shape") and y.shape == x.shape:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(x, y)
        ax.axhline(0, lw=1)
        ax.axvline(0, lw=1)
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(equation)
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        return outfile
    else:
        raise ValueError("Evaluated expression did not produce a y(x) array.")

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

    st.divider()
    st.subheader("Optional: Graph (PNG)")
    want_graph = st.checkbox("Create graph (y = f(x)) as PNG", value=False)
    eq_guess = guess_equation(problem_text) or ""
    equation_to_plot = ""
    x_min = -5.0
    x_max = 5.0
    points = 1000
    if want_graph:
        equation_to_plot = st.text_input("Equation (e.g., y = x^2 - 1)", value=eq_guess)
        cols = st.columns(3)
        with cols[0]:
            x_min = st.number_input("x_min", value=-5.0)
        with cols[1]:
            x_max = st.number_input("x_max", value=5.0)
        with cols[2]:
            points = st.number_input("points", min_value=100, max_value=5000, value=1000, step=100)

# =========================
# MAIN
# =========================
if st.button("🚀 Solve & Generate (HTML + optional Graph PNG)", use_container_width=True):
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
    st.download_button("⬇️ Download HTML", data=html_str, file_name=html_filename, mime="text/html", use_container_width=True)

    if want_graph:
        if not equation_to_plot.strip():
            st.warning("Graphing enabled, but no equation provided. Please enter something like 'y = x^2 - 1'.")
        else:
            with st.spinner("Plotting equation as PNG..."):
                out_dir = Path("animated_exports"); out_dir.mkdir(exist_ok=True)
                png_path = out_dir / (stem + "-graph.png")
                try:
                    plot_equation_png(equation_to_plot, png_path, float(x_min), float(x_max), int(points))
                    st.image(str(png_path), caption=f"Graph: {equation_to_plot}", use_container_width=True)
                    st.download_button("⬇️ Download Graph PNG", data=png_path.read_bytes(), file_name=png_path.name, mime="image/png", use_container_width=True)
                except Exception as e:
                    st.warning("Could not plot the equation. See details below.")
                    st.exception(e)

    with st.expander("Show raw JSON from GPT"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
