from flask import Flask, render_template, request, g
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
import random
import sqlite3
from urllib.parse import urlparse, parse_qs
try:
    from google_play_scraper import reviews, Sort
except ImportError:
    pass
import re
from datetime import datetime
app = Flask(__name__)

# Load local model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  source TEXT, 
                  text_snippet TEXT, 
                  result TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_history(source, text_snippet, result):
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        snippet = text_snippet[:100] + "..." if len(text_snippet) > 100 else text_snippet
        c.execute("INSERT INTO history (timestamp, source, text_snippet, result) VALUES (?, ?, ?, ?)",
                  (timestamp, source, snippet, result))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

def get_history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, source, text_snippet, result FROM history ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        conn.close()
        return [{"time": r[0], "source": r[1], "text": r[2].capitalize() if r[2] else "", "result": r[3].capitalize()} for r in rows]
    except:
        return []

def get_history_aggregates():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT source, result FROM history")
        rows = c.fetchall()
        conn.close()
        
        pos = 0
        neg = 0
        neu = 0
        
        for r in rows:
            source = r[0]
            res = r[1].lower() if r[1] else ""
            
            if source == "Single Text":
                if "positive" in res:
                    pos += 1
                elif "negative" in res:
                    neg += 1
                elif "neutral" in res:
                    neu += 1
            else:
                pos_match = re.search(r'\+([0-9]+)', res)
                neg_match = re.search(r'\-([0-9]+)', res)
                neu_match = re.search(r'\=([0-9]+)', res)
                
                if pos_match: pos += int(pos_match.group(1))
                if neg_match: neg += int(neg_match.group(1))
                if neu_match: neu += int(neu_match.group(1))
                
        return {"positive": pos, "negative": neg, "neutral": neu}
    except Exception as e:
        print(f"Agg Error: {e}")
        return {"positive": 0, "negative": 0, "neutral": 0}

# ---------------- API HELPER ----------------
def analyze_with_api(text, api_key):
    try:
        API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                top_result = max(results[0], key=lambda x: x['score'])
                label = top_result['label'].lower()
                if 'positive' in label: return 'positive'
                if 'negative' in label: return 'negative'
                return 'neutral'
    except Exception as e:
        print(f"API Error: {e}")
    return None

def predict_sentiment(text, api_key):
    if api_key and api_key.strip():
        api_result = analyze_with_api(text, api_key.strip())
        if api_result: return (api_result, "Hugging Face Cloud API")
        
    # Local fallback
    try:
        review_vector = vectorizer.transform([text])
        
        # If the input contains zero words the model recognizes, it's inherently neutral
        if review_vector.nnz == 0:
            return ("neutral", "Local Model")
            
        # If the model is not very confident (e.g., between 45% and 55%), flag as neutral
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(review_vector)[0]
            if max(probs) < 0.55:
                return ("neutral", "Local Model")
                
        prediction = model.predict(review_vector)[0]
        return (str(prediction).lower(), "Local Model")
    except:
        return ("neutral", "Local Model")

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    history = get_history()
    agg = get_history_aggregates()
    return render_template("index.html", history=history, agg=agg)

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form.get('review', '')
    api_key = request.form.get('api_key', '')

    if not review.strip():
        return render_template("index.html", result="Enter a review", history=get_history(), agg=get_history_aggregates())

    sentiment, source_model = predict_sentiment(review, api_key)
    
    if "positive" in sentiment:
        result_text = f"Positive 😊 (Powered by {source_model})"
    elif "negative" in sentiment:
        result_text = f"Negative 😡 (Powered by {source_model})"
    else:
        result_text = f"Neutral 😐 (Powered by {source_model})"

    save_history("Single Text", review, result_text)
    
    return render_template("index.html", result=result_text, history=get_history(), agg=get_history_aggregates())

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    api_key = request.form.get('api_key', '')

    if not file:
        return "No file uploaded"

    try:
        df = pd.read_csv(file)
    except:
        return "Error reading CSV"

    if 'review' not in df.columns:
        return "CSV must contain 'review' column"

    results = []
    
    # Take up to 50 rows for time safety
    for review in df['review'].head(50):
        review = str(review)
        if review.strip() == "":
            continue
        sentiment, _ = predict_sentiment(review, api_key)
        results.append(sentiment)

    if len(results) == 0:
        return "No valid data in CSV"

    positive = results.count("positive")
    negative = results.count("negative")
    neutral = results.count("neutral")
    
    save_history("CSV Upload", f"Analyzed {len(results)} rows", f"+{positive} | -{negative} | ={neutral}")

    return render_template("result.html", positive=positive, negative=negative, neutral=neutral)

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.form.get('url', '').strip()
    api_key = request.form.get('api_key', '')
    
    if not url:
        return "No URL provided"
    
    scraped_texts = []
    
    if "play.google.com" in url:
        try:
            parsed_url = urlparse(url)
            app_id = parse_qs(parsed_url.query).get('id', [None])[0]
            if app_id:
                app_reviews, _ = reviews(app_id, count=20, sort=Sort.NEWEST)
                scraped_texts = [r['content'] for r in app_reviews]
            else:
                return "Could not extract App ID from Google Play URL."
        except Exception as e:
            return f"Error scraping Google Play: {e}"
    elif "google.com/maps" in url:
        return "<div style='padding: 2rem; text-align: center; font-family: sans-serif;'><h2>Cannot Scrape Google Maps</h2><p>Google Maps requires an API key to scrape. Please try a <b>Google Play Store</b> link or standard article instead.</p><a href='/'>Go Back</a></div>"
    else:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            scraped_texts = [p.get_text() for p in paragraphs if len(p.get_text().split()) > 5][:10]
        except Exception as e:
            return f"Error scraping website: {e}"

    if not scraped_texts:
        return "Could not find any readable text/reviews on that page."

    results = []
    for text in scraped_texts:
        sentiment, _ = predict_sentiment(text, api_key)
        results.append(sentiment)

    if len(results) == 0:
        return "No valid text to analyze from URL"

    positive = results.count("positive")
    negative = results.count("negative")
    neutral = results.count("neutral")

    save_history("URL Scan", url, f"+{positive} | -{negative} | ={neutral}")

    return render_template("result.html", positive=positive, negative=negative, neutral=neutral)

if __name__ == "__main__":
    app.run(debug=True)