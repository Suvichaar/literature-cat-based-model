import io
import base64
from typing import Optional

import streamlit as st

# Azure Document Intelligence SDK
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# DOCX writer
from docx import Document
from docx.shared import Pt

# =========================
# CONFIG / SECRETS
# =========================
# Prefer st.secrets if available; fallback to the provided constants

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return default

AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY = get_secret("AZURE_DI_KEY")

# =========================
# UI
# =========================
st.set_page_config(page_title="PDF → DOCX (Azure Document Intelligence)", page_icon="📄", layout="centered")
st.title("📄 PDF → DOCX with Azure Document Intelligence (Read)")
st.caption("Upload a PDF → Azure DI (prebuilt-read) extracts text → Download a .docx")

with st.expander("Settings", expanded=False):
    add_page_breaks = st.checkbox("Insert page breaks between PDF pages", value=True)
    include_confidence = st.checkbox("Append line confidence (debug)", value=False)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

# =========================
# Helpers
# =========================
@st.cache_resource(show_spinner=False)
def make_client(endpoint: str, key: str):
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        raise RuntimeError("Azure SDK not installed. Run: pip install azure-ai-documentintelligence")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def analyze_pdf_bytes(client: DocumentIntelligenceClient, pdf_bytes: bytes):
    """
    Calls DI prebuilt-read. Supports both modern (document=stream) and older (base64Source) invocation.
    """
    try:
        # Newer SDKs generally accept a stream as 'document'
        poller = client.begin_analyze_document(model_id="prebuilt-read", document=pdf_bytes)
    except TypeError:
        # Fallback: some versions require base64 payload
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            analyze_request={"base64Source": b64}
        )
    return poller.result()

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
            # Page header
            doc.add_heading(f"Page {idx+1}", level=2)

            # Some SDK versions expose 'lines' on page; others require reading from 'paragraphs'
            if getattr(page, "lines", None):
                for ln in page.lines:
                    text = ln.content or ""
                    if show_conf and hasattr(ln, "spans") and ln.spans:
                        # confidence is usually on words/spans; this is a rough approximation
                        try:
                            confs = [getattr(s, "confidence", None) for s in ln.spans if getattr(s, "confidence", None) is not None]
                            if confs:
                                text += f"  [conf~{sum(confs)/len(confs):.2f}]"
                        except Exception:
                            pass
                    if text.strip():
                        doc.add_paragraph(text)
            else:
                # Fallback: use result.paragraphs filtered by page number
                paras = []
                for p in getattr(result, "paragraphs", []) or []:
                    if getattr(p, "spans", None):
                        # If any span falls on this page, include it
                        if any(getattr(sp, "offset", None) is not None for sp in p.spans):
                            paras.append(p.content)
                if paras:
                    for p in paras:
                        doc.add_paragraph(p)
                else:
                    # Ultimate fallback: write the global content
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
            client = make_client(AZURE_DI_ENDPOINT, AZURE_DI_KEY)
        except Exception as e:
            st.error(f"Failed to create Azure DI client: {e}")
            st.stop()

        pdf_bytes = uploaded.read()

        with st.spinner("Analyzing with Azure Document Intelligence (prebuilt-read)..."):
            try:
                result = analyze_pdf_bytes(client, pdf_bytes)
            except Exception as e:
                st.error(f"Azure DI analyze failed: {e}")
                st.stop()

        # Build DOCX
        with st.spinner("Building DOCX..."):
            try:
                docx_bytes = result_to_docx_bytes(result, insert_page_breaks=add_page_breaks, show_conf=include_confidence)
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

        # Optional quick preview of first ~2k chars of extracted text
        with st.expander("Preview extracted text (first ~2,000 chars)"):
            preview = getattr(result, "content", "")
            st.text(preview[:2000] + ("..." if len(preview) > 2000 else ""))

else:
    st.info("Upload a PDF to begin.")

