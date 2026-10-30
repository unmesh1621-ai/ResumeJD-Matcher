"""
Streamlit demo: upload / paste a resume and a job description, get the fit
score from all four trained encoders plus matched and missing skills.

Run: streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import read_document, score_pair  # noqa: E402

st.set_page_config(page_title="Resume–JD Matcher", layout="wide")
st.title("Resume–Job Description Matcher")
st.caption("ICT 4442 mini project — Siamese MLP / 1D-CNN / BiLSTM / fine-tuned SBERT encoders, "
           "trained on real resume–JD pairs. Scores are cosine similarities; the decision "
           "threshold for each model was tuned on the validation split.")

col_r, col_j = st.columns(2)
with col_r:
    st.subheader("Resume")
    upload = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    resume_text = st.text_area("…or paste resume text", height=250)
with col_j:
    st.subheader("Job description")
    jd_text = st.text_area("Paste the job description", height=330)

if upload is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getbuffer())
    resume_text = read_document(tmp.name)

if st.button("Score match", type="primary", disabled=not (resume_text.strip() and jd_text.strip())):
    with st.spinner("Encoding with all four models…"):
        results, explanation = score_pair(resume_text, jd_text)

    st.subheader("Fit scores")
    df = pd.DataFrame(results)
    df["decision"] = df["match"].map({True: "✅ match", False: "❌ no match"})
    st.dataframe(df[["model", "score", "threshold", "decision", "test_auc"]]
                 .rename(columns={"test_auc": "model test AUC"}).style.format(
                     {"score": "{:.3f}", "threshold": "{:.3f}", "model test AUC": "{:.3f}"}),
                 hide_index=True, use_container_width=True)

    st.subheader("Why — skills (explainability layer)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Keyword overlap", f"{explanation['keyword_overlap_ratio']:.0%}")
    c2.markdown("**Matched skills**\n\n" + (", ".join(explanation["matched_skills"]) or "—"))
    c3.markdown("**Missing skills (gap)**\n\n" + (", ".join(explanation["missing_skills"]) or "—"))
