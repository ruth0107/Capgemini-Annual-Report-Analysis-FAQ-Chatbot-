import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
import logging
import os
import json
import time
import requests 
from io import BytesIO
from pypdf import PdfReader 
import asyncio # Used locally to run synchronous requests in a Streamlit context

# --- Semantic Dependencies ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# -----------------------------

# --- Configuration & Setup ---
st.set_page_config(
    page_title="PDF RAG Chatbot (TF-IDF)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress minor warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Define chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 250 # <-- INCREASED OVERLAP FOR BETTER CONTEXT BRIDGING

# --- Define Default PDF Filename ---
DEFAULT_PDF_FILENAME = "Annual-Report-2024-25.pdf"

# Gemini API Configuration
# Reads the key securely from the environment variable GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# Initialize resources
@st.cache_resource
def setup_nlp_tools():
    """Initializes and caches NLTK resources."""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return stop_words, lemmatizer
    except LookupError:
        # Fallback in case NLTK components haven't been downloaded (requires manual nltk.download calls)
        st.error("NLTK resources (stopwords, wordnet) may be missing. Ensure they are downloaded.")
        return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nlp_tools()


# --- Text Processing Functions ---

def basic_clean(text):
    """Preprocess: convert to lower, remove newlines, digits, punctuation, special chars"""
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_words(text, do_lemmatize=True):
    """Word Tokenize, clean, remove stopwords, and lemmatize."""
    text = basic_clean(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

@st.cache_data
def get_pdf_text_and_chunks(file_data, file_name):
    """
    Extracts text from PDF (BytesIO buffer), cleans it, and splits it into chunks.
    """
    try:
        # 1. Reset file pointer and create BytesIO buffer from raw data
        pdf_reader = PdfReader(BytesIO(file_data))
        
        raw_text = ""
        for page in pdf_reader.pages:
            # Safely extract text, using "" for pages that yield None
            raw_text += page.extract_text() or ""
        
        # 2. Simple Chunking Logic (using raw text for context retrieval)
        # We clean the text for processing/indexing, but keep the original for retrieval
        cleaned_text = raw_text 
        
        chunks = []
        i = 0
        while i < len(cleaned_text):
            chunk = cleaned_text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP 
            
        if not chunks:
            st.error("Could not extract any meaningful text from the PDF.")
            return None, None, None
            
        return raw_text, chunks, file_name
    except Exception as e:
        # Generic error handling for PDF reading issues
        st.error(f"Error reading PDF file: {e}")
        return None, None, None

@st.cache_data
def create_tfidf_index(chunks):
    """
    Creates the TF-IDF Vectorizer and Matrix from the text chunks.
    """
    # Clean chunks for TF-IDF training
    clean_chunks = [" ".join(preprocess_words(chunk, do_lemmatize=True)) for chunk in chunks]

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
        max_df=0.90, 
        min_df=5
    )
    
    tfidf_matrix = vectorizer.fit_transform(clean_chunks)
    
    return vectorizer, tfidf_matrix, clean_chunks


def get_best_match(query, chunks, vectorizer, tfidf_matrix):
    """
    Retrieves the most relevant chunk using TF-IDF and Cosine Similarity (Semantic Match).
    Includes a keyword boost for high-value terms like 'revenue' or 'total' to improve precision.
    """
    
    # 1. Clean and Transform the Query
    query_tokens = preprocess_words(query, do_lemmatize=True)
    clean_query = " ".join(query_tokens)
    if not clean_query:
        return None, 0.0
    
    # Check if query words exist in the vectorizer's vocabulary
    if not any(token in vectorizer.vocabulary_ for token in query_tokens):
         # Fallback check, though the low score should handle this mostly
         return None, 0.0

    query_vector = vectorizer.transform([clean_query])
    
    # 2. Calculate Cosine Similarity
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 3. Apply Keyword Boost (to solve the 'Manpower' problem)
    
    # Define keywords that signal financial facts or specific metrics
    high_value_keywords = ['revenue', 'profit', 'total', 'manpower', 'employees', 'capital', 'divestments', 'subsidiary']
    
    # Create a simple boost array (1.0 for no boost, 1.1 for a boost)
    boost_array = np.ones(len(chunks))
    for i, chunk in enumerate(chunks):
        if any(keyword in chunk.lower() for keyword in high_value_keywords):
            boost_array[i] = 1.15 # Apply a slight, controlled boost (15%)
            
    boosted_scores = cosine_scores * boost_array

    # 4. Find Best Match using boosted score
    best_match_index = np.argmax(boosted_scores)
    best_score = cosine_scores[best_match_index] # Report the true cosine similarity score
    best_chunk = chunks[best_match_index]
            
    return best_chunk, best_score

# --- LLM Generation Function (Synchronous) ---

def generate_answer_with_llm(user_query, context_sentence, retrieval_score):
    """
    Calls the Gemini API using the synchronous 'requests' library.
    """
    
    if not API_KEY:
        return "**API Key Missing Error:** The RAG Chatbot requires the Gemini API Key to generate conversational answers."

    # 1. Define the system instruction (persona and rules)
    system_instruction = (
        "You are an expert financial analyst chatbot providing concise, conversational answers "
        "based STRICTLY on the provided context, which is a segment retrieved from an Annual Report. "
        "If the context is irrelevant or insufficient, politely state that you cannot answer the question "
        "with the information provided. Do not use external knowledge. "
        "Crucial Rule: If the user asks for a 'Consolidated' figure, ensure the retrieved text explicitly contains 'Consolidated' before answering. If it only contains 'Standalone' or 'Company' figures, you must refuse to answer the 'Consolidated' question, even if the numbers look similar."
    )
    
    # 2. Define the user prompt (question + context)
    prompt = (
        f"Based on the retrieved text below, answer the user's question conversationally.\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"CONTEXT (from report, relevance score: {retrieval_score:.3f}): {context_sentence}\n\n"
        f"CONVERSATIONAL ANSWER:"
    )

    # 3. Construct the API Payload
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": 0.2} 
    }

    # 4. API Call with Exponential Backoff
    max_retries = 3
    retry_delay = 1 

    for attempt in range(max_retries):
        try:
            full_api_url = f"{API_URL}?key={API_KEY}"
            
            response = requests.post(
                full_api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() 
            
            result = response.json()
            
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: LLM response missing text.')
            
            # Format the output for the user
            retrieval_info = (
                f"\n\n---\n"
                f"**🤖 RAG Grounding:**\n"
                f"**Relevance Score (Semantic Similarity):** {retrieval_score:.3f}\n"
                f"**Source Context:** *{context_sentence}*"
            )
            return text + retrieval_info

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2 
            else:
                return f"Sorry, I couldn't connect to the LLM. Network/API Error: {e}"
        except Exception as e:
            return f"Sorry, an unexpected error occurred during processing: {e}"
            
    return "Failed to generate response after multiple retries."


# --- Streamlit UI and Logic ---

def main():
    
    st.title("🧠 RAG Chatbot: Capegemini FAQ's")
    st.markdown("Uploaded PDF is used to build a semantic index for the whole document. Answers are based on **meaningful chunks** from the full report.")

    # --- Sidebar for PDF Upload ---
    with st.sidebar:
        st.header("1. Data Source")
        
        file_data = None
        file_name = None
        uploaded_file = None 

        # 1. Check for default file existence (local file path)
        if os.path.exists(DEFAULT_PDF_FILENAME):
            try:
                st.success(f"Default file **{DEFAULT_PDF_FILENAME}** found. Loading automatically.")
                # Read the file content into a byte buffer
                with open(DEFAULT_PDF_FILENAME, "rb") as f:
                    file_data = f.read()
                file_name = DEFAULT_PDF_FILENAME
            except Exception as e:
                st.error(f"Error reading local file: {e}")
                
        # 2. Fallback to manual upload if automatic load fails or file is missing
        if file_data is None:
            st.warning(f"Default file **{DEFAULT_PDF_FILENAME}** not found or failed to load.")
            uploaded_file = st.file_uploader(
                "Upload Capgemini Annual Report PDF", 
                type="pdf",
                key="pdf_uploader", 
                help=f"If you name your file exactly '{DEFAULT_PDF_FILENAME}' and place it next to this script, it will load automatically next time."
            )
            if uploaded_file is not None:
                file_data = uploaded_file.getvalue()
                file_name = uploaded_file.name
                
        st.markdown(f"**Chunk Size:** {CHUNK_SIZE} characters | **Overlap:** {CHUNK_OVERLAP} characters")
        
        st.markdown("---")
        st.header("2. Topics Covered")
        
        # --- UI IMPROVEMENT: Use expander and custom styling ---
        with st.expander("**Optimized Report Sections**", expanded=True):
            st.markdown(
                """
                ### Common Queries & Testing Examples

                **1. Financial Performance**
                - What was the total **Consolidated Profit** for 2024-25?
                - What was the **revenue from operations** for the year?
                
                **2. Human Resources & Governance**
                - What is the total **manpower figure** as of March 31, 2025?
                - How many women employees received benefits under the **Maternity Benefit Act**?
                
                **3. Sustainability & CSR**
                - Explain the goal of the **'Mission Million Trees'** project.
                - What technology is used in the **Battery Energy Storage Solution (BESS)**?
                """
            )
        # --- END UI IMPROVEMENT ---


    # --- Data Loading and Indexing ---
    data_source_key = file_name # Use file name for robust cache key when loading from disk
    
    if file_data is None:
        st.warning("Please upload a PDF or ensure the default file is present to begin the analysis.")
        return

    # Process PDF into Chunks
    # This function extracts text and creates chunks based on size parameters
    raw_text, chunks, processed_file_name = get_pdf_text_and_chunks(file_data, file_name)
    if chunks is None:
        return 

    # Create Semantic Index
    with st.spinner(f"Indexing {processed_file_name} into {len(chunks)} chunks for semantic retrieval..."):
        # This builds the TF-IDF vectorizer and matrix
        vectorizer, tfidf_matrix, clean_chunks = create_tfidf_index(chunks)
    
    st.success(f"Indexed {processed_file_name} into {len(chunks)} chunks. Ready for chat.")

    if vectorizer is None:
        return
        
    # Check for API Key presence after data is loaded
    if not API_KEY:
        st.error(
            "🛑 **API Key Not Found!**\n\n"
            "To use the RAG Chatbot, you must set your Gemini API key as an environment variable named `GEMINI_API_KEY`."
        )
        return

    # Chat Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"The full report ({processed_file_name}) has been indexed. Ask me about **consolidated revenue** or **CSR projects**."})

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_query = st.chat_input("Ask a question about the full report...")

    if user_query:
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # RAG Pipeline Execution
        with st.chat_message("assistant"):
            # 1. RETRIEVE: Get the best context chunk using TF-IDF/Cosine Similarity
            best_chunk, best_score = get_best_match(user_query, chunks, vectorizer, tfidf_matrix)
            
            # Use a threshold to filter out poor matches (score is generally low for TF-IDF)
            if best_chunk is None or best_score < 0.1: 
                 st.error("I couldn't find a strongly relevant chunk in the report (Score too low). Please try rephrasing with specific keywords.")
                 st.session_state.messages.append({"role": "assistant", "content": "I couldn't find a strongly relevant chunk in the report. Please try rephrasing with specific keywords."})
                 return

            # 2. GENERATE: Pass context to LLM
            with st.spinner(f"Retrieving context (Score: {best_score:.3f})... Generating conversational answer with Gemini LLM..."):
                response_text = generate_answer_with_llm(
                    user_query, 
                    best_chunk, 
                    best_score
                )
                st.markdown(response_text)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
