import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend import answer

st.title("RAG Application")
query = st.text_input("Ask a question about the syllabus:")
if st.button("Chat"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        response = answer(query)
        st.markdown(f"**Answer:** {response}")
