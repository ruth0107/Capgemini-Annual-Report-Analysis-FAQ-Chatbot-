# ==============================================================================
# NLP Mini Project (Part A): Annual Report Analysis (Capgemini 2024-25)
# FILE 1/3: Core NLP Processing Script
# Instructions:
# 1. Place your Capgemini AR PDF in the same folder as this script.
# 2. Update the 'PDF_PATH' variable below if the filename is different.
# 3. Run this script once: 'python project_script.py'
# 4. It will generate three CSV files and two PNG images required for your report.
# ==============================================================================

import os
import re
import pdfplumber
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
import numpy as np
from nltk.stem import WordNetLemmatizer
import logging, warnings

# --- Configuration ---
# !!! IMPORTANT: CHANGE THIS FILENAME IF YOUR PDF IS NAMED DIFFERENTLY !!!
PDF_PATH = "Capgemini_Annual_Report_2024-25.pdf"

OUTPUT_CSV_SENTENCES = "capgemini_sentences_raw.csv"
OUTPUT_CSV_SENTIMENT = "capgemini_sentences_with_sentiment.csv"
OUTPUT_CSV_TOPICS = "capgemini_sentences_with_topics.csv"
OUTPUT_IMAGE_WORDCLOUD = "wordcloud_capgemini.png"
OUTPUT_IMAGE_BARCHART = "barchart_top_words.png"
NUM_TOPICS = 10 # Required number of topics

# --- Setup ---
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Download NLTK resources if not present (important for local runs)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except:
    nltk.download('omw-1.4', quiet=True)


# --- Global Resources and Helper Functions ---
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    """Task 1: Import pdf and read all pages"""
    all_text = []
    if not os.path.exists(pdf_path):
        print(f"🛑 Error: PDF file not found at {pdf_path}. Please place the file in the script directory.")
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        raw_text = "\n".join(all_text)
        print(f"✅ Extracted {len(raw_text):,} characters from PDF.")
        return raw_text
    except Exception as e:
        print(f"🛑 Error reading PDF: {e}")
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
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    if do_lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

def sentence_sentiment(text):
    """Task 4: Calculate sentiment for each sentence (TextBlob)"""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def get_topic_distribution_for_doc(bow, model):
    dist = model.get_document_topics(bow, minimum_probability=0.0)
    vector = np.zeros(model.num_topics)
    for topic_id, prob in dist:
        vector[topic_id] = prob
    return vector

# ==============================================================================
# === MAIN WORKFLOW EXECUTION ===
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting Capgemini NLP Mini Project (Part A) ---")

    # 1) Import pdf and read all pages
    raw_text = extract_text_from_pdf(PDF_PATH)
    if raw_text is None:
        exit()

    # 2) Save it into a dataframe (Sentence level)
    sentences = sent_tokenize(raw_text)
    df_sents = pd.DataFrame({"sentence": sentences})
    df_sents.index.name = "sent_id"
    df_sents.to_csv(OUTPUT_CSV_SENTENCES, index=True)
    print(f"✅ Saved {len(df_sents):,} raw sentences to {OUTPUT_CSV_SENTENCES}")

    # 3 & 4) Preprocess & Calculate Sentiment
    df_sents['clean_sentence'] = df_sents['sentence'].apply(basic_clean)
    df_sents[['sentiment_polarity', 'sentiment_subjectivity']] = df_sents['sentence'].apply(
        lambda s: pd.Series(sentence_sentiment(s))
    )
    
    print("\n--- Sentiment Analysis Summary ---")
    print(f"Average Polarity (Tone): {df_sents['sentiment_polarity'].mean():.4f}")
    df_sents.to_csv(OUTPUT_CSV_SENTIMENT, index=True)
    print(f"✅ Saved sentence-level sentiment to {OUTPUT_CSV_SENTIMENT}")

    # 5 & 6) Word Tokenize, Preprocess, and Wordcloud
    all_tokens = preprocess_words(raw_text, do_lemmatize=True)
    
    freq = Counter(all_tokens)
    top_20 = freq.most_common(20)

    print("\n--- Top 20 Most Frequent Words ---")
    for i, (word, count) in enumerate(top_20):
        print(f"{i+1:2d}. {word:12s}: {count}")

    # Generate and Save Wordcloud (Image for report)
    wc_text = " ".join(all_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(wc_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Wordcloud: Capgemini Annual Report")
    plt.savefig(OUTPUT_IMAGE_WORDCLOUD)
    plt.close()
    print(f"✅ Saved Wordcloud image to {OUTPUT_IMAGE_WORDCLOUD}")

    # Generate and Save Bar Chart (Image for report)
    words, counts = zip(*top_20)
    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color='#00796B')
    plt.xlabel("Frequency")
    plt.title("Top 20 Frequent Words")
    plt.savefig(OUTPUT_IMAGE_BARCHART)
    plt.close()
    print(f"✅ Saved Bar Chart image to {OUTPUT_IMAGE_BARCHART}")

    # 7) Prepare DTM/Corpus for LDA
    sent_docs_lda = df_sents['sentence'].apply(lambda x: preprocess_words(x, do_lemmatize=True))
    sent_docs_lda = sent_docs_lda[sent_docs_lda.apply(len) >= 3]

    dictionary = corpora.Dictionary(sent_docs_lda)
    dictionary.filter_extremes(no_below=5, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in sent_docs_lda]
    
    # 8) Build Topic Modelling (LDA)
    print(f"\n--- Running Gensim LDA for {NUM_TOPICS} Topics ---")
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=42,
        passes=10,
        iterations=400,
        alpha='auto',
    )

    topic_dists = [get_topic_distribution_for_doc(bow, lda_model) for bow in corpus]
    topic_df = pd.DataFrame(topic_dists, columns=[f"topic_{i+1}" for i in range(NUM_TOPICS)])

    df_topics = df_sents.loc[sent_docs_lda.index].reset_index(drop=True).copy()
    df_topics.index.name = 'sent_id'
    df_topics = df_topics.join(topic_df)

    df_topics.to_csv(OUTPUT_CSV_TOPICS, index=True)
    print(f"✅ Saved sentences with topic distributions to {OUTPUT_CSV_TOPICS}")

    print("\n--- Project Execution Complete. Check your folder for the output files. ---")
