import tempfile
import streamlit as st

from pdfloader import load_pdf
from vectorstore import build_vectorstore
from chains import create_chain, llm

st.set_page_config(
    page_title = "Research Paper Explainer"
)

st.title("🧾 Research Paper Explainer")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type = ["pdf"]
)

if uploaded_file :

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as f :

        f.write(uploaded_file.read())

        pdf_path = f.name

    with st.spinner("Processing...") :

        docs = load_pdf(pdf_path)

        vector_db = build_vectorstore(docs)

        chain = create_chain(vector_db)

        result = chain.invoke("Summarize this research paper.")

        st.markdown(result)