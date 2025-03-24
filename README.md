# NewsInsight AI 📰✨  
*AI-Powered News Summarization & Sentiment Analysis Tool*

## Table of Contents
- [Features](#features-)
- [Tech Stack](#tech-stack-)
- [Installation](#installation-)
- [Usage](#usage-)
- [API Reference](#api-reference-)
- [Project Structure](#project-structure-)
- [Examples](#examples-)
- [Contributing](#contributing-)
- [License](#license-)

## Features ✨
- **Automated News Collection**: Scrape latest articles from Google News
- **Smart Summarization**: Generate concise summaries using FLAN-T5
- **Sentiment Analysis**: 3-model ensemble (DistilBERT + Flair + VADER)
- **Topic Extraction**: RAKE + KeyBERT for keyphrase identification
- **Multilingual TTS**: Convert insights to Hindi speech
- **Comparative Reports**: Analyze coverage differences across sources

## Tech Stack 🛠️
| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| NLP Models | Hugging Face Transformers |
| Web Scraping | BeautifulSoup + SERPAPI |
| Text Processing | NLTK |
| Audio | gTTS |

## Installation 🚀
```bash
git clone https://github.com/yourusername/NewsInsight-AI.git
cd NewsInsight-AI
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
echo "SERPAPI_KEY=your_key_here" > .env
