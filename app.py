import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192")  # ✅ works


# Streamlit UI
st.set_page_config(page_title="Speech Summarizer", layout="wide")
st.title("🗣️ Multilingual Speech Summarizer & Translator")

st.markdown("Upload a speech or paste it below, and get a concise, multilingual summary with bullet points.")

# Input methods
speech_text = st.text_area("📄 Paste Speech Text", height=300)

uploaded_file = st.file_uploader("📁 Upload a Speech File (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        speech_text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        speech_text = ""
        for page in pdf_reader.pages:
            speech_text += page.extract_text()


# Language selection
language = st.selectbox("🌐 Choose Summary Language", ["English", "Hindi", "French", "German", "Spanish"])

if st.button("Generate Summary") and speech_text:
    with st.spinner("Summarizing..."):

        # Text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = splitter.split_documents([Document(page_content=speech_text)])

        # Map Prompt Template
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Please summarize the below speech:
Speech:`{text}`
Summary:
"""
        )

        # Final Reduce Prompt Template
        final_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Provide the final summary of the entire speech with these important points.
Add a Motivation Title, start the precise summary with an introduction, and provide the summary in numbered points.

Speech: {text}
"""
        )

        # Load summarization chain
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=final_prompt,
            verbose=False
        )

        # Run chain
        final_summary = summary_chain.run(docs)

        # Translate if not English
        if language != "English":
            translation_template = PromptTemplate(
                input_variables=['speech', 'language'],
                template="""
Write a summary of the following speech:
Speech:{speech}
Translate the precise summary to {language}
"""
            )
            translator = LLMChain(llm=llm, prompt=translation_template)
            final_summary = translator.run({'speech': final_summary, 'language': language})

        # Output
        st.subheader("📌 Summary Output")
        st.markdown(final_summary)

else:
    st.info("Please paste a speech or upload a file and click 'Generate Summary'.")

