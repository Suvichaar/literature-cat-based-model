import io
import base64
from typing import Optional, Any

import streamlit as st

# Azure Document Intelligence SDK
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# Some SDK variants expose a typed request model
try:
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest  # type: ignore
except Exception:
    AnalyzeDocumentRequest = None  # type: ignore

# DOCX writer
from docx import Document
from docx.shared import Pt

# =========================
# CONFIG / SECRETS
# =========================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return default

AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY = get_secret("AZURE_DI_KEY")

st.set_page_config(page_title="PDF → DOCX (Azure Document Intelligence)", page_icon="📄", layout="centered")
st.title("📄 PDF → DOCX with Azure Document Intelligence (Read)")
st.caption("Upload a PDF → Azure DI (prebuilt-read) extracts text → Download a .docx")

with st.expander("Settings", expanded=False):
    add_page_breaks = st.checkbox("Insert page breaks between PDF pages", value=True)
    include_confidence = st.checkbox("Append line confidence (debug)", value=False)

# If secrets are not set, show inline inputs so you can still run locally
if not AZURE_DI_ENDPOINT or not AZURE_DI_KEY:
    st.info("Azure DI endpoint/key not found in st.secrets. Enter them below for this session.")
    AZURE_DI_ENDPOINT = st.text_input("AZURE_DI_ENDPOINT", AZURE_DI_ENDPOINT or "", placeholder="https://<resourcename>.cognitiveservices.azure.com/")
    AZURE_DI_KEY = st.text_input("AZURE_DI_KEY", AZURE_DI_KEY or "", type="password")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

# =========================
# Helpers
# =========================
@st.cache_resource(show_spinner=False)
def make_client(endpoint: str, key: str):
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        raise RuntimeError("Azure SDK not installed. Run: pip install azure-ai-documentintelligence")
    if not endpoint or not key:
        raise RuntimeError("Missing Azure DI endpoint or key.")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def analyze_pdf_bytes(client: Any, pdf_bytes: bytes):
    """
    Calls Azure Document Intelligence 'prebuilt-read' across multiple SDK variants.
    Tries (in order):
      A) document=pdf_bytes, content_type="application/pdf"         (newer SDKs)
      B) body={"base64Source": "..."}                               (older SDKs)
      C) body=AnalyzeDocumentRequest(bytes_source=pdf_bytes)        (some mid SDKs)
    Raises the last error if all fail.
    """
    last_err = None

    # A) Newer SDK signature
    try:
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            document=pdf_bytes,
            content_type="application/pdf",
        )
        return poller.result()
    except Exception as e:
        last_err = e  # keep and try next

    # B) Older SDK signature (base64Source in body)
    try:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        poller = client.begin_analyze_document(
            "prebuilt-read",
            body={"base64Source": b64},
        )
        return poller.result()
    except Exception as e:
        last_err = e

    # C) Typed request model (some versions)
    try:
        if AnalyzeDocumentRequest is not None:
            req = AnalyzeDocumentRequest(bytes_source=pdf_bytes)  # type: ignore
            poller = client.begin_analyze_document(
                model_id="prebuilt-read",
                body=req,
                content_type="application/pdf",
            )
            return poller.result()
    except Exception as e:
        last_err = e

    # If we’re here, surface the most recent error
    raise last_err

def result_to_docx_bytes(result, insert_page_breaks: bool = True, show_conf: bool = False) -> bytes:
    """
    Convert DI 'prebuilt-read' result into a simple DOCX.
    - Writes per-line text (preserves reading order per page)
    - Optional page breaks between pages
    """
    doc = Document()

    # Basic style tweaks (optional)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    if not getattr(result, "pages", None):
        # Fallback: if we only have 'content', dump it
        doc.add_paragraph(getattr(result, "content", "") or "No content found.")
    else:
        for idx, page in enumerate(result.pages):
            doc.add_heading(f"Page {idx+1}", level=2)

            # Prefer page.lines if present
            if getattr(page, "lines", None):
                for ln in page.lines:
                    text = ln.content or ""
                    if show_conf and hasattr(ln, "spans") and ln.spans:
                        try:
                            confs = [getattr(s, "confidence", None) for s in ln.spans if getattr(s, "confidence", None) is not None]
                            if confs:
                                text += f"  [conf~{sum(confs)/len(confs):.2f}]"
                        except Exception:
                            pass
                    if text.strip():
                        doc.add_paragraph(text)
            else:
                # Fallback: use result.paragraphs filtered by page number (best-effort)
                paras = []
                for p in getattr(result, "paragraphs", []) or []:
                    if getattr(p, "spans", None):
                        if any(getattr(sp, "offset", None) is not None for sp in p.spans):
                            paras.append(p.content)
                if paras:
                    for p in paras:
                        doc.add_paragraph(p)
                else:
                    doc.add_paragraph(getattr(result, "content", "") or "")

            if insert_page_breaks and idx < len(result.pages) - 1:
                doc.add_page_break()

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# =========================
# MAIN
# =========================
if uploaded is not None:
    if not uploaded.name.lower().endswith(".pdf"):
        st.error("Please upload a PDF file.")
    else:
        # Initialize client
        try:
            client = make_client(AZURE_DI_ENDPOINT or "", AZURE_DI_KEY or "")
        except Exception as e:
            st.error(f"Failed to create Azure DI client: {e}")
            st.stop()

        pdf_bytes = uploaded.read()
        if not pdf_bytes:
            st.error("Uploaded file is empty. Please re-upload the PDF.")
            st.stop()

        with st.spinner("Analyzing with Azure Document Intelligence (prebuilt-read)..."):
            try:
                result = analyze_pdf_bytes(client, pdf_bytes)
            except Exception as e:
                st.error(f"Azure DI analyze failed: {e}")
                st.stop()

        # Build DOCX
        with st.spinner("Building DOCX..."):
            try:
                docx_bytes = result_to_docx_bytes(
                    result,
                    insert_page_breaks=add_page_breaks,
                    show_conf=include_confidence
                )
            except Exception as e:
                st.error(f"Failed to create DOCX: {e}")
                st.stop()

        st.success("Done! Your DOCX is ready.")
        st.download_button(
            label="⬇️ Download .docx",
            data=docx_bytes,
            file_name=(uploaded.name.rsplit(".", 1)[0] + ".docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        with st.expander("Preview extracted text (first ~2,000 chars)"):
            preview = getattr(result, "content", "")
            st.text(preview[:2000] + ("..." if len(preview) > 2000 else ""))

else:
    st.info("Upload a PDF to begin.")
