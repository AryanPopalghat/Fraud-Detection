from flask import Flask, request, render_template
from newspaper import Article
from openai import OpenAI  # Ensure you import OpenAI correctly
import re
import time
import tldextract
import requests
from urllib.parse import urlparse
from serpapi.google_search import GoogleSearch
from bs4 import BeautifulSoup
import os
from goose3 import Goose
from exa_py import Exa
from dotenv import load_dotenv
from pathlib import Path



g = Goose()
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
app = Flask(__name__)
port = int(os.environ.get("PORT", 5005))

def listToString(s):
    return "".join(s)

# Configure APIs with your keys
openai_api_key = os.getenv('openai_api_key')
serpapi_api_key = os.getenv('serpapi_api_key')
exa = Exa(os.getenv('exa'))
# List of reputable news sources
reputable_sources = [
    "nytimes.com", "wsj.com", "washingtonpost.com", "reuters.com", "bloomberg.com",
    "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com", "foxnews.com",
    "bbc.com", "theguardian.com", "economist.com", "npr.org", "apnews.com", "allindiaradio.gov.in","ptinews.com",
    "rewariyasat.com","etvbharat.com","news18.com","dailyexcelsior.com","dnaindia.com","deccanherald.com","indiatvnews.com",
    "news18.com/amp/","igod.gov.in","thehindu.com","hindustantimes.com","newindpress.com","ranchiexpress.com/news/","rediff.com/news/","rediff.com",
    "thestatesman.net","hugedomains.com","indianexpress.com","telegraphindia.com","indiatimes.com","tribuneindia.com","theweek.in","theprint.in","en.wikipedia.org","ndtv.com"
]

# Function to generate similar headlines using OpenAI's GPT-3.5 Turbo
def generate_similar_headlines(headline):
    try:
        client = OpenAI(
        # This is the default and can be omitted
        api_key=openai_api_key,
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating similar headlines and start and end each headline with term HeadLineWord like HeadLineWord word1 word2 HeadLineWord."},
                {"role": "user", "content": f"Here is the original headline: '{headline}'. Generate 5 similar headlines that capture the essence of the news article."}
            ],
            max_tokens=150,
            n=1,  # Generate one completion with multiple headlines
            stop=None,
            temperature=0.7,
        )
        
        # Extract headlines
        similar_headlines = response.choices[0].message.content
        
        # Filter out empty lines and ensure there are exactly 5 headlines
        similar_headlines = [headline.strip() for headline in similar_headlines.split("\n") if headline.strip()]
        if len(similar_headlines) > 5:
            similar_headlines = similar_headlines[:5]
        return similar_headlines
    except Exception as e:
        print(f"Error generating similar headlines: {e}")
        return []

# Function to search for headlines using SerpAPI
def search_for_headlines(query):
    params = {
        "api_key": serpapi_api_key,
        "engine": "google",
        "q": query,
        "num": 10,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

# Function to search for headlines using Exa Api
def search_using_exa(query):
    result = exa.search_and_contents(
    query,
    type="neural",
    use_autoprompt=True,
    num_results=10,
    text=True,
    )
    return result.results

# Function to extract source domain from URL
def extract_source(article_url):
    ext = tldextract.extract(article_url)
    return f"{ext.domain}.{ext.suffix}"

# Function to find sentences between words
def find_sentences_between_word(text, starting_word, ending_word):
    matches = [
        (match.start(), match.end())
        for match in re.finditer(
            re.escape(starting_word) + r"(.*?)" + re.escape(ending_word),
            text,
            re.DOTALL,
        )
    ]
    sentences = []
    for start_index, end_index in matches:
        sentence_between = text[
            start_index + len(starting_word): end_index - len(ending_word)
        ].strip()
        if sentence_between:
            sentences.append(sentence_between)
    return sentences

# Function to check news credibility
def check_news_credibility(article_url, threshold=5):
    article = g.extract(url=article_url)
    headline = listToString(article.title)
    article_url = listToString(article_url)
    source = extract_source(article_url)
    similar_headlines = "".join(generate_similar_headlines(headline))
    search_queries = find_sentences_between_word(similar_headlines, 'HeadLineWord', 'HeadLineWord')

    related_articles_count_serp = 0
    related_articles_count_exa = 0
    related_articles_count = 0
    website_traffic_scores = []
    start_time = time.time()
    # while time.time() - start_time < 30:  # 1/2 minute
    #     for query in search_queries:
    #         search_results = search_for_headlines(query)
    #         if search_results:
    #             related_articles_count_serp += len(search_results)

    #     if related_articles_count_serp >= threshold:
    #         break
        
    while time.time() - start_time < 60:  # 1 minute
        for query in search_queries:
            search_results = search_using_exa(query)
            if search_results:
                related_articles_count_exa += len(search_results)

        if related_articles_count_exa >= threshold:
            break
    
    related_articles_count = (related_articles_count_serp + related_articles_count_exa) / 50
    print('relate articles count_serp=',related_articles_count_serp, '   related articles count exa=', related_articles_count_exa,'  /n')
    source_credibility_score = 1.0 if source in reputable_sources else 0.5
    
    credibility_score = (0.4 * related_articles_count + 0.6 * source_credibility_score) 

    return credibility_score, source_credibility_score, related_articles_count

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/check', methods=['POST'])
def check():
    article_url = request.form['article_url']
    credibility_score, source_credibility_score, related_articles_count = check_news_credibility(article_url)
    return render_template(
        'index1.html', 
        credibility_score=credibility_score, 
        source_credibility_score=source_credibility_score, 
        related_articles_count=related_articles_count,
        formula="credibility_score = (0.4 * related_articles_count + 0.6 * source_credibility_score) * 100 % ",
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
