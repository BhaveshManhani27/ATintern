import random
from deep_translator import GoogleTranslator  # Changed library
from gtts import gTTS
from utils import load_models, scrape_google_news, generate_summary, ensemble_sentiment, extract_topics_rake, find_topic_overlap, generate_comparison_and_impact, generate_sentiment_summary

models = load_models()

def text_to_speech(summary, output_audio_file="output_speech.mp3"):
    # Step 1: Translate the English text to Hindi using deep-translator
    try:
        translated_summary = GoogleTranslator(source='en', target='hi').translate(summary)
    except Exception as e:
        print(f"Translation failed: {e}")
        translated_summary = summary  # Fallback to English if translation fails

    # Step 2: Convert the translated text to speech
    tts = gTTS(text=translated_summary, lang='hi')  # Set language to Hindi
    tts.save(output_audio_file)
    print(f"Audio saved to {output_audio_file}")

# ... rest of your api.py code remains unchanged ...
def analyze_news(company_name: str):
    try:
        # Scrape Google News
        articles = scrape_google_news(company_name)
        if not articles:
            raise Exception("No articles found for the given company name.")

        # Summarize articles
        for article in articles:
            article["Summary"] = generate_summary(article["Full_Content"], models["summary_pipeline"])

        # Randomly select 2 articles for analysis
        random_articles = random.sample(articles, 2)

        # Perform sentiment analysis
        article1_sentiment = ensemble_sentiment(random_articles[0]["Full_Content"], models["distilbert_pipeline"], models["flair_classifier"], models["vader_analyzer"])
        article2_sentiment = ensemble_sentiment(random_articles[1]["Full_Content"], models["distilbert_pipeline"], models["flair_classifier"], models["vader_analyzer"])

        # Extract topics
        for article in random_articles:
            article["Topics"] = extract_topics_rake(article["Full_Content"], models["rake"])

        # Find topic overlap
        topic_overlap = find_topic_overlap(random_articles[0], random_articles[1], models["sentence_model"], models["rake"])

        # Generate comparisons and impacts
        coverage_differences = generate_comparison_and_impact(random_articles[0]["Full_Content"], random_articles[1]["Full_Content"], models["summary_pipeline"])

        # Generate sentiment summary
        sentiment_summary = generate_sentiment_summary(
            random_articles[0]["Full_Content"],
            random_articles[1]["Full_Content"],
            article1_sentiment,
            article2_sentiment,
            models["summary_pipeline"]
        )

        # Convert summary to speech
        text_to_speech(sentiment_summary)

        # Prepare final output
        final_output = {
            "Company": company_name,
            "Articles": [
                {
                    "Title": random_articles[0]["Title"],
                    "Summary": random_articles[0]["Summary"],
                    "Sentiment": article1_sentiment,
                    "Topics": random_articles[0]["Topics"]
                },
                {
                    "Title": random_articles[1]["Title"],
                    "Summary": random_articles[1]["Summary"],
                    "Sentiment": article2_sentiment,
                    "Topics": random_articles[1]["Topics"]
                }
            ],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": {
    "Positive": (1 if article1_sentiment == "POSITIVE" else 0) + (1 if article2_sentiment == "POSITIVE" else 0),
    "Negative": (1 if article1_sentiment == "NEGATIVE" else 0) + (1 if article2_sentiment == "NEGATIVE" else 0),
    "Neutral": (1 if article1_sentiment == "NEUTRAL" else 0) + (1 if article2_sentiment == "NEUTRAL" else 0)
                },
                "Coverage Differences": coverage_differences,
                "Topic Overlap": topic_overlap
            },
            "Final Sentiment Analysis": sentiment_summary,
            "Audio": "[Play Hindi Speech]"
        }

        return final_output

    except Exception as e:
        # Re-raise the exception so that the Gradio interface can handle it.
        raise Exception(str(e))

        # Summarize articles
        for article in articles:
            article["Summary"] = generate_summary(article["Full_Content"], models["summary_pipeline"])

        # Randomly select 2 articles for analysis
        random_articles = random.sample(articles, 2)

        # Perform sentiment analysis
        article1_sentiment = ensemble_sentiment(random_articles[0]["Full_Content"], models["distilbert_pipeline"], models["flair_classifier"], models["vader_analyzer"])
        article2_sentiment = ensemble_sentiment(random_articles[1]["Full_Content"], models["distilbert_pipeline"], models["flair_classifier"], models["vader_analyzer"])

        # Extract topics
        for article in random_articles:
            article["Topics"] = extract_topics_rake(article["Full_Content"], models["rake"])

        # Find topic overlap
        topic_overlap = find_topic_overlap(random_articles[0], random_articles[1], models["sentence_model"], models["rake"])

        # Generate comparisons and impacts
        coverage_differences = generate_comparison_and_impact(random_articles[0]["Full_Content"], random_articles[1]["Full_Content"], models["summary_pipeline"])

        # Generate sentiment summary
        sentiment_summary = generate_sentiment_summary(
            random_articles[0]["Full_Content"],
            random_articles[1]["Full_Content"],
            article1_sentiment,
            article2_sentiment,
            models["summary_pipeline"]
        )

        # Convert summary to speech
        text_to_speech(sentiment_summary)

        # Prepare final output
        final_output = {
            "Company": company_name,
            "Articles": [
                {
                    "Title": random_articles[0]["Title"],
                    "Summary": random_articles[0]["Summary"],
                    "Sentiment": article1_sentiment,
                    "Topics": random_articles[0]["Topics"]
                },
                {
                    "Title": random_articles[1]["Title"],
                    "Summary": random_articles[1]["Summary"],
                    "Sentiment": article2_sentiment,
                    "Topics": random_articles[1]["Topics"]
                }
            ],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": {
    "Positive": (1 if article1_sentiment == "POSITIVE" else 0) + (1 if article2_sentiment == "POSITIVE" else 0),
    "Negative": (1 if article1_sentiment == "NEGATIVE" else 0) + (1 if article2_sentiment == "NEGATIVE" else 0),
    "Neutral": (1 if article1_sentiment == "NEUTRAL" else 0) + (1 if article2_sentiment == "NEUTRAL" else 0)
                },
                "Coverage Differences": coverage_differences,
                "Topic Overlap": topic_overlap
            },
            "Final Sentiment Analysis": sentiment_summary,
            "Audio": "[Play Hindi Speech]"
        }

        return final_output

    except Exception as e:
        # Re-raise the exception so that the Gradio interface can handle it.
        raise Exception(str(e))
