# ==============================================================================
# NLP Mini Project (Part A): Capgemini Web Dashboard - LOCAL FILE MODE
# This version loads a static PDF file and includes advanced visualizations.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pdfplumber
import gensim
from gensim import corpora
import logging, warnings
import io

# ----------------------------------------------------------------------
# FIX: SET_PAGE_CONFIG MOVED TO BE THE ABSOLUTE FIRST STREAMLIT COMMAND
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Annual Report Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ----------------------------------------------------------------------

# --- Configuration and File Path ---
LOCAL_PDF_PATH = "Annual-Report-2024-25.pdf" # Static file path
NUM_TOPICS = 10 
COMPANY_NAME = "Capgemini Technology Services India Limited"

# --- Setup and Global Resources ---
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure Matplotlib uses a non-interactive backend for Streamlit
plt.rcParams.update({'figure.max_open_warning': 0}) 

@st.cache_resource
def setup_nltk_resources():
    """Download necessary NLTK packages once and initialize resources."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = setup_nltk_resources()


# --- Topic Interpretation Function ---
def get_topic_interpretation(topic_id, keywords_str):
    """Provides a human-readable title for the LDA topics based on keywords."""
    keywords = set(re.split(r'[,\s]+', keywords_str.lower()))
    
    if any(k in keywords for k in ['financial', 'statement', 'tax', 'asset']):
        return "Financial Reporting, Assets & Tax"
    
    if any(k in keywords for k in ['director', 'board', 'committee', 'executive']):
        return "Corporate Governance & Board Structure"
        
    if any(k in keywords for k in ['service', 'technology', 'solution', 'client']):
        return "Client Services & Digital Strategy"
        
    if any(k in keywords for k in ['employee', 'gratuity', 'scheme', 'talent']):
        return "Human Capital & Employee Schemes"
        
    if any(k in keywords for k in ['risk', 'control', 'audit']):
        return "Risk Management & Internal Control"
        
    if any(k in keywords for k in ['share', 'equity', 'parent', 'subsidiary']):
        return "Shareholding & Group Structure"
        
    if 'revenue' in keywords and 'currency' in keywords:
        return "Revenue & Foreign Currency Risk"
        
    return "General Operations/Strategy"


# --- Helper Functions ---

@st.cache_data
def extract_text_from_pdf_path(pdf_path):
    """Task 1: Import pdf and read all pages from a static file path."""
    if not os.path.exists(pdf_path):
        st.error(f"🛑 Error: PDF file not found at {pdf_path}. Please ensure the file is in the project folder.")
        return None
        
    all_text = []
    try:
        with open(pdf_path, 'rb') as fp:
             with pdfplumber.open(fp) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
        raw_text = "\n".join(all_text)
        return raw_text
    except Exception as e:
        st.error(f"🛑 Error reading PDF content: {e}")
        return None

def basic_clean(text):
    """Task 3: Preprocessing base function"""
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_words(text, do_lemmatize=True):
    """Task 5: Word Tokenize and preprocess"""
    text = basic_clean(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def sentence_sentiment(text):
    """Task 4: Calculate sentiment for each sentence (TextBlob)"""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# --- Main Analysis Function ---
@st.cache_data(show_spinner="Running comprehensive NLP analysis (Sentiment, Frequency, and Topic Modeling)...")
def run_nlp_pipeline(raw_text, num_topics):
    """Executes the full NLP workflow on the extracted text."""
    
    # --- 2) Document and Sentence level DataFrames ---
    sentences = sent_tokenize(raw_text)
    df_sents = pd.DataFrame({"sentence": sentences})
    df_sents.index.name = "sent_id"

    # --- 3) Preprocess (Apply basic cleaning) ---
    df_sents['clean_sentence'] = df_sents['sentence'].apply(basic_clean)

    # --- 4) Sentiment Analysis (TextBlob) ---
    df_sents[['sentiment_polarity', 'sentiment_subjectivity']] = df_sents['sentence'].apply(
        lambda s: pd.Series(sentence_sentiment(s))
    )

    # --- 5) Word Tokenize and preprocess (Document-level tokens) ---
    all_tokens = preprocess_words(raw_text, do_lemmatize=True)
    
    # --- 6) Frequent word and Wordcloud data ---
    freq = Counter(all_tokens)
    top_tokens = freq.most_common(50)
    
    # --- 7) & 8) Topic Modelling (LDA with Gensim) ---
    sent_docs_lda = df_sents['sentence'].apply(lambda x: preprocess_words(x, do_lemmatize=True))
    sent_docs_lda = sent_docs_lda[sent_docs_lda.apply(len) >= 3] # Filter sentences with too few meaningful words

    dictionary = corpora.Dictionary(sent_docs_lda)
    dictionary.filter_extremes(no_below=5, no_above=0.8) # Filter extreme words
    corpus = [dictionary.doc2bow(text) for text in sent_docs_lda]

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        iterations=400,
        alpha='auto',
    )
    
    # Extract topics and keywords
    topics_keywords = []
    for idx in range(num_topics):
        top_words = lda_model.show_topic(idx, topn=10)
        keywords = ', '.join([word for word, prob in top_words])
        
        # Calculate topic weight in the whole corpus
        topic_words_df = pd.DataFrame(lda_model.show_topic(idx, topn=100), columns=['word', 'weight'])
        total_topic_weight = topic_words_df['weight'].sum()
        
        topics_keywords.append({
            "Topic ID": idx + 1,
            "Keywords": keywords,
            "Total Weight": total_topic_weight
        })
    df_topics_keywords = pd.DataFrame(topics_keywords)
    
    # Calculate topic distribution per sentence
    def get_topic_distribution_for_doc(bow, model):
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        vector = np.zeros(model.num_topics)
        for topic_id, prob in dist:
            vector[topic_id] = prob
        return vector

    topic_dists = [get_topic_distribution_for_doc(bow, lda_model) for bow in corpus]
    topic_df = pd.DataFrame(topic_dists, columns=[f"topic_{i+1}" for i in range(num_topics)])
    
    df_topics_sents = df_sents.loc[sent_docs_lda.index].reset_index(drop=True)
    df_topics_sents = df_topics_sents.join(topic_df)

    return df_sents, all_tokens, top_tokens, df_topics_keywords, df_topics_sents


# --- Streamlit UI Main Block ---

st.title(f"🤖 NLP Analysis: {COMPANY_NAME}")
st.markdown(f"**Document:** {LOCAL_PDF_PATH} (Loaded from directory)")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Analysis Controls")
    st.info(f"PDF document is static: **{LOCAL_PDF_PATH}**")
    
    NUM_TOPICS = st.slider(
        "Number of Topics (LDA)", 
        min_value=5, 
        max_value=20, 
        value=10, 
        step=1,
        help="The number of latent topics the model will try to identify."
    )
    
    st.markdown("---")
    st.caption("The analysis runs automatically on load or setting change.")

# --- Main Content Area ---

# Load and process the static PDF file
raw_text = extract_text_from_pdf_path(LOCAL_PDF_PATH)
file_name = LOCAL_PDF_PATH.replace(".pdf", "")

if raw_text:
    try:
        df_sents, all_tokens, top_tokens, df_topics_keywords, df_topics_sents = run_nlp_pipeline(raw_text, NUM_TOPICS)

        st.success(f"Analysis Complete! Processed {len(raw_text):,} characters and {len(df_sents):,} sentences.")
        
        # --- Tabs for Results ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "Sentiment Summary", 
            "Word Frequency & Cloud", 
            "Topic Modeling (LDA)", 
            "Download Data"
        ])

        # --- Tab 1: Sentiment Summary (NEW VISUALS) ---
        with tab1:
            st.header("Sentiment Analysis Overview")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sentences", f"{len(df_sents):,}")
            col2.metric("Average Polarity (Tone)", f"{df_sents['sentiment_polarity'].mean():.4f}", help="Close to 1.0 is positive, close to -1.0 is negative.")
            col3.metric("Average Subjectivity", f"{df_sents['sentiment_subjectivity'].mean():.4f}", help="Close to 1.0 is subjective/opinion-based, close to 0.0 is objective/factual.")

            st.markdown("---")
            st.subheader("Sentiment Distribution ")
            
            # Sentiment Histogram (NEW VISUAL)
            fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
            df_sents['sentiment_polarity'].hist(bins=20, ax=ax_hist, color='#4CAF50', edgecolor='black', alpha=0.7)
            ax_hist.axvline(df_sents['sentiment_polarity'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Polarity')
            ax_hist.set_title("Distribution of Sentence Polarity Scores")
            ax_hist.set_xlabel("Polarity Score (-1.0 to +1.0)")
            ax_hist.set_ylabel("Frequency (Number of Sentences)")
            ax_hist.legend()
            st.pyplot(fig_hist)
            
            st.caption("The histogram shows the overall neutral tone (scores clustered near zero) with a slight skew toward positive scores (right of the mean).")

        # --- Tab 2: Word Frequency & Cloud ---
        with tab2:
            st.header("Key Term Analysis")

            # Word Cloud
            col_wc, col_bar = st.columns(2)
            
            with col_wc:
                st.subheader("Word Cloud")
                wc_text = " ".join(all_tokens)
                wordcloud = WordCloud(
                    width=1000, 
                    height=500, 
                    background_color="white", 
                    collocations=False, 
                    max_words=100
                ).generate(wc_text)
                
                fig_wc, ax_wc = plt.subplots(figsize=(15, 8))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title(f"Wordcloud: {file_name}", fontsize=20)
                st.pyplot(fig_wc)

            # Top 20 Bar Chart
            with col_bar:
                st.subheader("Top 20 Frequent Words")
                top_20 = top_tokens[:20]
                words, counts = zip(*top_20)
                
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                ax_bar.barh(words[::-1], counts[::-1], color='#3a506b')
                ax_bar.set_xlabel("Frequency")
                ax_bar.set_title("Top 20 Frequent Words")
                st.pyplot(fig_bar)

        # --- Tab 3: Topic Modeling (LDA) (NEW VISUAL) ---
        with tab3:
            st.header(f"Latent Dirichlet Allocation ({NUM_TOPICS} Topics)")
            
            # Apply Topic Interpretation
            df_topics_keywords['Interpretation'] = df_topics_keywords['Topic ID'].apply(
                lambda i: get_topic_interpretation(i, df_topics_keywords.loc[df_topics_keywords['Topic ID'] == i, 'Keywords'].iloc[0])
            )
            
            st.subheader("Topic Keywords and Interpretation")
            
            # Topic Distribution Bar Chart (NEW VISUAL)
            fig_topic_dist, ax_topic_dist = plt.subplots(figsize=(12, 6))
            df_topics_keywords.sort_values(by='Total Weight', ascending=False, inplace=True)
            
            ax_topic_dist.barh(
                df_topics_keywords['Interpretation'][::-1], 
                df_topics_keywords['Total Weight'][::-1], 
                color='#FF8C00' # Orange theme
            )
            ax_topic_dist.set_title("Overall Importance of Identified Topics in the Report")
            ax_topic_dist.set_xlabel("Cumulative Weight of Topic Keywords (LDA)")
            plt.tight_layout()
            st.pyplot(fig_topic_dist)
            st.caption("This chart visually compares the overall dominance (weight) of each topic across the entire report's vocabulary.")
            
            st.markdown("---")
            st.subheader("Detailed Topic Composition")
            st.dataframe(df_topics_keywords[['Topic ID', 'Interpretation', 'Keywords']], hide_index=True, use_container_width=True)

            st.subheader("Sentence-Topic Distribution (Sample)")
            st.dataframe(
                df_topics_sents[['sentence'] + [f"topic_{i+1}" for i in range(NUM_TOPICS)]].head(10),
                use_container_width=True,
                height=500
            )
        
        # --- Tab 4: Download Data ---
        with tab4:
            st.header("Download Processed Data")
            
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_sentiment = convert_df_to_csv(df_sents.drop(columns=['clean_sentence']))
            st.download_button(
                label="Download Sentiment Data (CSV)",
                data=csv_sentiment,
                file_name=f'{file_name}_sentiment.csv',
                mime='text/csv',
                help="Includes raw sentence text, polarity, and subjectivity scores."
            )

            csv_topics = convert_df_to_csv(df_topics_sents)
            st.download_button(
                label="Download Topic Distribution Data (CSV)",
                data=csv_topics,
                file_name=f'{file_name}_topics.csv',
                mime='text/csv',
                help=f"Includes sentence text and probability distribution across {NUM_TOPICS} topics."
            )
    
    except Exception as e:
        # Catch errors from the NLP pipeline itself
        st.error(f"An error occurred during NLP processing. Please check the PDF content. Error: {e}")

else:
    # This block is reached if os.path.exists() failed above
    st.info(f"Waiting for the required PDF file: **{LOCAL_PDF_PATH}** to be placed in the directory.")
