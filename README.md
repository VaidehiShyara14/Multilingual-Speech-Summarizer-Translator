# 🗣️ Multilingual Speech Summarizer & Translator
An interactive web application to summarize speeches in multiple languages using advanced large language models. Upload a speech text or PDF, or paste the text directly, and get a concise summary with bullet points and multilingual translation support.

---

## Features

- Upload speech files in `.txt` or `.pdf` format or paste speech text directly.
- Summarizes long speeches using map-reduce strategy for efficient processing.
- Supports multilingual summaries in English, Hindi, French, German, and Spanish.
- Uses Groq’s LLaMA3 model via LangChain for high-quality summarization and translation.
- Provides output with motivational titles, introductions, and numbered summary points.

---

## Tech Stack

- **Python**  
- **Streamlit** – for building the web interface  
- **LangChain** – LLM chaining and prompt management  
- **Groq API (ChatGroq)** – LLaMA3-70B model integration  
- **PyPDF2** – PDF text extraction  
- **dotenv** – Environment variable management  

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/multilingual-speech-summarizer.git
   cd multilingual-speech-summarizer
