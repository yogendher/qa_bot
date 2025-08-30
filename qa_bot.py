import streamlit as st
from transformers import pipeline
import PyPDF2

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

st.title("QA Bot - Copy/Paste or PDF Upload")
st.write("You can either paste text manually or upload a PDF, then ask questions!")

# Option to choose input method
input_method = st.radio("Select input method:", ("Paste Text", "Upload PDF"))

context = ""

# --- Option 1: Paste Text ---
if input_method == "Paste Text":
    context = st.text_area("Paste your content here:")

# --- Option 2: Upload PDF ---
elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_list = [page.extract_text() for page in pdf_reader.pages]
        context = "\n".join(text_list)
        st.write("PDF uploaded successfully!")
        st.text_area("PDF Content Preview", value=context[:1000] + "...", height=200)  # preview first 1000 chars

# Ask question
question = st.text_input("Enter your question:")

if question and context:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])
