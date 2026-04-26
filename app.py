from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import os
import secrets
from ultimate_ensemble import create_ultimate_analyzer
import warnings
from aspect_analyze import analyze_healthcare_feedback, AspectBasedAnalyzer
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)

# Set secret key for sessions
app.secret_key = secrets.token_hex(32)

# ============================================
# STEP 1: LOAD YOUR ML MODEL FIRST
# ============================================
print("Loading healthcare sentiment model...")

ensemble = None
vectorizer = None

# Try to load model
model_path = 'models/healthcare_model_improved.pkl'

if os.path.exists(model_path):
    print(f"Found model at: {model_path}")
    try:
        model_data = joblib.load(model_path)
        if 'ensemble' in model_data:
            ensemble = model_data['ensemble']
        elif 'model' in model_data:
            ensemble = model_data['model']
        else:
            ensemble = model_data
        
        if 'vectorizer' in model_data:
            vectorizer = model_data['vectorizer']
        
        print("✅ ML Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if ensemble is None:
    print("No model found, using fallback mode")

# Initialize VADER once (only once)
sia = SentimentIntensityAnalyzer()

# ============================================
# STEP 2: CREATE ANALYZER (ONLY ONE!)
# ============================================
print("Creating enhanced healthcare analyzer...")
generalized_analyzer = create_ultimate_analyzer(ensemble, vectorizer)
print("Analyzer ready!")

# ============================================
# STEP 3: SENTIMENT PREDICTION FUNCTION
# ============================================

def predict_sentiment(text, api_key=None):
    """Generalized sentiment prediction that works for ANY review"""
    return generalized_analyzer.predict(text)


# ============================================
# DATABASE FUNCTIONS
# ============================================
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source TEXT,
        department TEXT,
        text_snippet TEXT,
        result TEXT,
        confidence REAL
    )''')
    
    c.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'department' not in columns:
        c.execute("ALTER TABLE history ADD COLUMN department TEXT DEFAULT 'general'")
    if 'confidence' not in columns:
        c.execute("ALTER TABLE history ADD COLUMN confidence REAL DEFAULT 0.0")
    
    conn.commit()
    conn.close()
    print("Database initialized")

def save_to_history(source, department, text_snippet, result, confidence):
    if source == "Single Text":
        try:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snippet = text_snippet[:100] + "..." if len(text_snippet) > 100 else text_snippet
            c.execute("""INSERT INTO history 
                         (timestamp, source, department, text_snippet, result, confidence) 
                         VALUES (?, ?, ?, ?, ?, ?)""",
                      (timestamp, source, department, snippet, result, confidence))
            conn.commit()
            conn.close()
            print(f"✅ Saved: {department} - {snippet[:50]}...")
            return True
        except Exception as e:
            print(f"Save error: {e}")
    return False

def get_history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, source, department, text_snippet, result FROM history ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "time": row[0],
                "source": row[1],
                "department": row[2] if row[2] else "general",
                "text": row[3],
                "result": row[4]
            })
        return history
    except Exception as e:
        print(f"History error: {e}")
        return []

def get_history_aggregates():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT result FROM history")
        rows = c.fetchall()
        conn.close()
        
        pos = neg = neu = 0
        for row in rows:
            result = row[0].lower()
            if 'positive' in result:
                pos += 1
            elif 'negative' in result:
                neg += 1
            else:
                neu += 1
        return {"positive": pos, "negative": neg, "neutral": neu}
    except:
        return {"positive": 0, "negative": 0, "neutral": 0}

def get_department_stats():
    """Get sentiment stats by department"""
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT department, result FROM history")
        rows = c.fetchall()
        conn.close()
        
        stats = {
            'doctor': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0},
            'nurse': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0},
            'general': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        }
        
        print(f"\n📊 Processing {len(rows)} records for department stats")
        
        for dept, result in rows:
            if not dept:
                dept = 'general'
            else:
                dept = dept.lower().strip()
            
            if dept == 'doctor' or dept == 'physician':
                dept_key = 'doctor'
            elif dept == 'nurse' or dept == 'nursing':
                dept_key = 'nurse'
            else:
                dept_key = 'general'
            
            stats[dept_key]['total'] += 1
            
            result_lower = result.lower() if result else ''
            if 'positive' in result_lower:
                stats[dept_key]['positive'] += 1
            elif 'negative' in result_lower:
                stats[dept_key]['negative'] += 1
            else:
                stats[dept_key]['neutral'] += 1
        
        print(f"📊 Department Stats Results:")
        for dept in ['doctor', 'nurse', 'general']:
            print(f"   {dept}: {stats[dept]['total']} records (P:{stats[dept]['positive']}, Neu:{stats[dept]['neutral']}, Neg:{stats[dept]['negative']})")
        
        return stats
        
    except Exception as e:
        print(f"Department stats error: {e}")
        return {
            'doctor': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0},
            'nurse': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0},
            'general': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        }

def clear_history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return True
    except:
        return False

# ============================================
# ROUTES
# ============================================
@app.route('/')
def home():
    history = get_history()
    agg = get_history_aggregates()
    dept_stats = get_department_stats()
    return render_template("index.html", history=history, agg=agg, dept_stats=dept_stats)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Patient Feedback Analysis with Aspect-Based Analysis"""
    review = request.form.get('review', '')
    department = request.form.get('department', 'general')
    
    if department not in ['doctor', 'nurse', 'general']:
        department = 'general'
    
    print(f"\n📝 Received: Dept={department}, Review={review[:50]}...")
    
    if not review.strip():
        history = get_history()
        agg = get_history_aggregates()
        dept_stats = get_department_stats()
        return render_template("index.html", result="Please enter a review", 
                             history=history, agg=agg, dept_stats=dept_stats)
    
    # Get aspect-based analysis
    try:
        aspect_result = analyze_healthcare_feedback(review)
    except:
        aspect_result = {'insights': [], 'summary': 'No aspect analysis available'}
    
    # Get sentiment prediction
    result_text, confidence = predict_sentiment(review)
    
    # Add aspect insights to result
    if aspect_result.get('insights'):
        insights_html = "<br><br><strong>📋 Actionable Insights:</strong><br>"
        for insight in aspect_result['insights'][:3]:
            insights_html += f"• {insight}<br>"
        result_text += insights_html
    
    # Save to history
    save_to_history("Single Text", department, review, result_text, confidence)
    
    # Get updated data
    history = get_history()
    agg = get_history_aggregates()
    dept_stats = get_department_stats()
    
    return render_template("index.html", result=result_text, 
                         history=history, agg=agg, dept_stats=dept_stats)

@app.route('/upload', methods=['POST'])
def upload():
    """Bulk CSV Analysis - Optimized for large files"""
    file = request.files.get('file')
    
    if not file:
        return "No file uploaded"
    
    try:
        # Read CSV in chunks for large files (optional)
        df = pd.read_csv(file)
        
        if 'review' not in df.columns:
            return "CSV must contain a 'review' column"
        
        # Clean data efficiently
        df['review'] = df['review'].astype(str).str.strip()
        df = df[df['review'] != ''].head(50)
        
        if len(df) == 0:
            return "No valid reviews found"
        
        # Vectorized apply (much faster than loop)
        # Method 1: Using lambda with apply
        df['prediction'] = df['review'].apply(lambda x: predict_sentiment(x)[0])
        
        # Method 2: Alternative using a helper function (cleaner)
        # def get_sentiment(text):
        #     return predict_sentiment(text)[0]
        # df['prediction'] = df['review'].apply(get_sentiment)
        
        # Extract sentiment category
        df['sentiment'] = df['prediction'].str.extract(r'(Positive|Negative|Neutral)')[0].str.lower()
        
        # Count using pandas (fast)
        counts = df['sentiment'].value_counts().to_dict()
        
        return render_template("result.html",
                             positive=counts.get('positive', 0),
                             negative=counts.get('negative', 0),
                             neutral=counts.get('neutral', 0))
    
    except Exception as e:
        # Remove original_df reference if it exists
        # The error was likely referencing an undefined variable
        return f"Error processing file: {str(e)}"

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.form.get('url', '').strip()
    
    if not url:
        return "No URL provided"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        paragraphs = soup.find_all("p")
        texts = [p.get_text().strip() for p in paragraphs if len(p.get_text().split()) > 5][:10]
        
        if not texts:
            return render_template("result.html", positive=0, negative=0, neutral=0, scan_details=[], source_url=url)
        
        scan_details = []
        sentiments = []
        
        for text in texts:
            result_text, confidence = predict_sentiment(text)
            scan_details.append({
                'text': text[:300] + "..." if len(text) > 300 else text,
                'sentiment': result_text,
                'confidence': confidence
            })
            
            if 'Positive' in result_text:
                sentiments.append('positive')
            elif 'Negative' in result_text:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        
        positive = sentiments.count('positive')
        negative = sentiments.count('negative')
        neutral = sentiments.count('neutral')
        
        return render_template("result.html", positive=positive, negative=negative, neutral=neutral, 
                             scan_details=scan_details, source_url=url)
        
    except Exception as e:
        return render_template("result.html", positive=0, negative=0, neutral=0, scan_details=[], source_url=url, error=str(e))

@app.route('/clear_history', methods=['POST'])
def clear_history_route():
    success = clear_history()
    return jsonify({"success": success})

@app.route('/department_stats')
def department_stats_page():
    stats = get_department_stats()
    return render_template("department_stats.html", stats=stats)

@app.route('/api/department_stats')
def api_department_stats():
    stats = get_department_stats()
    return jsonify(stats)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    init_db()
    
    print("\n" + "="*60)
    print("HEALTHCARE SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    print(f"ML Model: {'Loaded' if ensemble else 'Not loaded (using fallback)'}")
    print(f"Enhanced Analyzer: {'Active' if generalized_analyzer else 'Inactive'}")
    print(f"Secret Key: Set ✓")
    print("\nFeatures:")
    print("   Patient Feedback → SAVED to history")
    print("   Department stats → Available for filtering")
    print("   Enhanced Healthcare Lexicon → Active")
    print("\nServer: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)