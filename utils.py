import torch
import requests
import time
import random
import re
import json
import nltk
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, DistilBertTokenizer, \
    DistilBertForSequenceClassification
from flair.models import TextClassifier
from flair.data import Sentence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS

# Download NLTK data
nltk.download('punkt', download_dir='/root/nltk_data')
nltk.download('stopwords', download_dir='/root/nltk_data')
nltk.download('punkt_tab', download_dir='/root/nltk_data')
nltk.data.path.insert(0, '/root/nltk_data')

# Constants
SERPAPI_KEY = "dc6c7662159faf1677caf498f9b8a34e2dd38e69ef0dbbb5ef491a42375dc0c5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize models
def load_models():
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(DEVICE)
    summary_pipeline = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer,
                                device=0 if DEVICE == "cuda" else -1)

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    distilbert_pipeline = pipeline("sentiment-analysis", model=distilbert_model, tokenizer=distilbert_tokenizer)

    flair_classifier = TextClassifier.load('en-sentiment')
    vader_analyzer = SentimentIntensityAnalyzer()
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(sentence_model)
    rake = Rake()

    return {
        "summary_pipeline": summary_pipeline,
        "distilbert_pipeline": distilbert_pipeline,
        "flair_classifier": flair_classifier,
        "vader_analyzer": vader_analyzer,
        "sentence_model": sentence_model,
        "kw_model": kw_model,
        "rake": rake,
    }


# Helper functions
def get_random_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }


def extract_full_content(url):
    try:
        response = requests.get(url, headers=get_random_headers(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        full_content = " ".join([p.get_text() for p in paragraphs if p.get_text()]).strip()
        return full_content if full_content else None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def scrape_google_news(company_name, max_pages=5):
    articles = []
    base_url = "https://serpapi.com/search"

    for page in range(max_pages):
        if len(articles) >= 10:
            break

        params = {
            "q": f"{company_name} news",
            "tbm": "nws",
            "api_key": SERPAPI_KEY,
            "start": page * 10
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if "news_results" not in data:
            continue

        for article in data["news_results"]:
            if len(articles) >= 10:
                break

            title = article.get("title", "").strip()
            url = article.get("link", "").strip()
            full_content = extract_full_content(url)

            if title and full_content and url:
                articles.append({
                    "Title": title,
                    "Full_Content": full_content,
                    "URL": url
                })

        time.sleep(1)

    return articles


def generate_summary(text, summary_pipeline):
    if not text or len(text) < 50:
        return "Summary not available"
    prompt = f"Summarize: {text[:4000]}"
    summary = summary_pipeline(prompt, max_length=150, truncation=True)
    return summary[0]['generated_text']


def ensemble_sentiment(text, distilbert_pipeline, flair_classifier, vader_analyzer):
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"

    max_length = 512
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    sentiment_votes = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for chunk in text_chunks:
        distilbert_result = distilbert_pipeline(chunk)[0]['label']
        sentiment_votes[distilbert_result] += 1

        sentence = Sentence(chunk)
        flair_classifier.predict(sentence)
        flair_sentiment = sentence.labels[0].value.upper()
        sentiment_votes[flair_sentiment] += 1

        vader_score = vader_analyzer.polarity_scores(chunk)["compound"]
        if vader_score >= 0.05:
            sentiment_votes["POSITIVE"] += 1
        elif vader_score <= -0.05:
            sentiment_votes["NEGATIVE"] += 1
        else:
            sentiment_votes["NEUTRAL"] += 1

    return max(sentiment_votes, key=sentiment_votes.get)


def extract_topics_rake(text, rake, n_topics=3):
    if not text or len(text) < 50:
        return []
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:n_topics]
    topics = [phrase.replace(" ", "_") for phrase in ranked_phrases]
    return topics


def find_topic_overlap(article1, article2, sentence_model, rake):
    topics1 = extract_topics_rake(article1["Full_Content"], rake)
    topics2 = extract_topics_rake(article2["Full_Content"], rake)

    if not topics1 or not topics2:
        return {
            "Common Topics": [],
            "Unique Topics in Article 1": topics1,
            "Unique Topics in Article 2": topics2
        }

    embeddings1 = sentence_model.encode(topics1)
    embeddings2 = sentence_model.encode(topics2)

    if embeddings1.ndim == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1:
        embeddings2 = embeddings2.reshape(1, -1)

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    common_topics = []
    unique_topics_1 = list(topics1)
    unique_topics_2 = list(topics2)

    for i, topic1 in enumerate(topics1):
        for j, topic2 in enumerate(topics2):
            if similarity_matrix[i][j] > 0.6:
                common_topics.append(topic1)
                if topic1 in unique_topics_1:
                    unique_topics_1.remove(topic1)
                if topic2 in unique_topics_2:
                    unique_topics_2.remove(topic2)

    return {
        "Common Topics": common_topics,
        "Unique Topics in Article 1": unique_topics_1,
        "Unique Topics in Article 2": unique_topics_2
    }


def generate_comparison_and_impact(article1, article2, comparison_pipeline):
    coverage_differences = []
    for i in range(2):
        comparison_prompt = f"""
        You are a journalist analyzing news coverage. Compare two articles on the same topic.
        Ensure your response is **detailed (80-90 words), unique**, and clearly explains how the coverage differs.

        Article 1: {article1[:2500]}
        Article 2: {article2[:2500]}

        Provide a **thorough and unique comparison**.
        """
        comparison_text = generate_text(comparison_prompt, comparison_pipeline)

        impact_prompt = f"""
        You are a market analyst. Analyze the **impact** of the different coverage styles in two articles.
        Ensure your response is **detailed (80-90 words), unique**, and explains **public, investor, or business reactions**.

        Article 1: {article1[:2500]}
        Article 2: {article2[:2500]}

        Provide a **deep and unique impact analysis**.
        """
        impact_text = generate_text(impact_prompt, comparison_pipeline)

        coverage_differences.append({
            "Comparison": comparison_text,
            "Impact": impact_text
        })
    return coverage_differences


def generate_text(prompt, comparison_pipeline, max_length=300, temperature=0.8):
    response = comparison_pipeline(prompt, max_length=max_length, temperature=temperature, do_sample=True)
    return response[0]['generated_text'].strip()


def generate_sentiment_summary(article1_text, article2_text, article1_sentiment, article2_sentiment, summary_pipeline):
    prompt = f"""Generate a concise market impact summary based on sentiment analysis of these news articles. Follow these steps:

1. Compare sentiment scores: 
  - Article 1 Sentiment: {article1_sentiment} 
  - Article 2 Sentiment: {article2_sentiment}

2. Create final summary that:
  - Starts with overall sentiment trend
  - Mentions specific impactful aspects
  - Predicts likely market consequences
  - Uses cautious language for mixed signals
  - Keeps under 2 sentences

Example Format: "[Company] coverage shows [trend] sentiment. [Specific aspect] and [specific aspect] suggest [predicted impact]."

Now analyze:
{article1_text[:500]}
{article2_text[:500]}"""

    summary_text = generate_summary(prompt, summary_pipeline)
    return summary_text


def text_to_speech(summary, output_audio_file="output_speech.mp3"):
    # Step 1: Translate the English text to Hindi
    translator = Translator()
    try:
        translated_summary = translator.translate(summary, src='en', dest='hi').text
    except Exception as e:
        print(f"Translation failed: {e}")
        translated_summary = summary  # Fallback to English if translation fails

    # Step 2: Convert the translated text to speech
    tts = gTTS(text=translated_summary, lang='hi')  # Set language to Hindi
    tts.save(output_audio_file)


def get_company_name():
    max_attempts = 10000
    attempts = 0

    while attempts < max_attempts:
        company_name = input("Enter the company name (e.g., Tesla, Apple, etc.): ").strip()

        if not company_name:
            print("Error: Company name cannot be empty. Please try again.")
            attempts += 1
            continue

        if not re.match(r"^[A-Za-z0-9\s\-&]+$", company_name):
            print("Error: Invalid company name. Please use only letters, numbers, spaces, hyphens, or '&'.")
            attempts += 1
            continue

        return company_name

    print("Maximum attempts reached. Exiting.")
    exit()
