"""
BALANCED GENERALIZED HEALTHCARE SENTIMENT ANALYZER
"""

import re
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import os

class BalancedHealthcareAnalyzer:
    """
    Truly balanced sentiment analyzer with excellent neutral detection
    """
    
    def __init__(self, ml_model=None, vectorizer=None):
        self.sia = SentimentIntensityAnalyzer()
        self.ml_model = ml_model
        self.vectorizer = vectorizer
        # Healthcare-specific terms (ADD THIS)
        self.healthcare_positive = {
            'recovered', 'healed', 'improved', 'better', 'relieved', 'cured',
            'successful', 'effective', 'worked', 'helped', 'beneficial',
            'explained', 'listened', 'answered', 'clarified', 'understood',
            'compassionate', 'caring', 'attentive', 'thorough', 'knowledgeable'
        }

        self.healthcare_negative = {
            'worsened', 'complicated', 'failed', 'ineffective', 'unsuccessful',
            'ignored', 'interrupted', 'rushed', 'dismissive', 'arrogant',
            'negligent', 'incompetent', 'careless', 'neglect', 'abandoned'
        }

        # Add to positive_words (merge with existing)
        self.positive_words.update(self.healthcare_positive)

        # Add to negative_words (merge with existing)
        self.negative_words.update(self.healthcare_negative)
        # NEUTRAL PHRASES (HIGHEST PRIORITY)
        self.neutral_phrases = [
            'nothing special', 'not special', 'nothing great', 'nothing bad',
            'nothing good', 'nothing terrible', 'nothing horrible',
            'neither good nor bad', 'not good not bad', 'so-so',
            'nothing to write home about', 'nothing to complain about',
            'nothing exceptional', 'nothing outstanding', 'nothing remarkable',
            'just okay', 'just fine', 'just average', 'just normal',
            'could be better could be worse', 'not impressed not disappointed',
            'no strong feelings', 'no complaints no praise', 'neutral experience',
            'nothing positive nothing negative', 'mixed feelings'
        ]
        
        # Neutral indicators
        self.neutral_indicators = {
            'special', 'average', 'normal', 'standard', 'regular', 'typical',
            'ordinary', 'common', 'usual', 'routine', 'basic', 'simple',
            'acceptable', 'adequate', 'decent', 'fair', 'moderate', 'reasonable',
            'fine', 'ok', 'okay', 'alright', 'passable', 'tolerable', 'mediocre'
        }
        
        # Strong positive
        self.strong_positive = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'perfect', 'exceptional', 'incredible', 'awesome', 'brilliant',
            'life-saving', 'saved my life', 'best ever', 'could not be better'
        }
        
        # Strong negative
        self.strong_negative = {
            'terrible', 'horrible', 'awful', 'dreadful', 'atrocious',
            'appalling', 'disgraceful', 'shocking', 'horrific', 'nightmare',
            'worst ever', 'never again', 'complete waste', 'absolutely terrible'
        }
        
        # Balanced sentiment lexicons
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'perfect', 'exceptional', 'incredible', 'awesome', 'brilliant',
            'great', 'good', 'nice', 'pleasant', 'satisfactory',
            'compassionate', 'caring', 'attentive', 'professional', 'thorough',
            'helpful', 'friendly', 'comfortable', 'efficient', 'saved',
            'recovered', 'grateful', 'satisfied', 'healed', 'improved',
            'supportive', 'understanding', 'patient', 'kind', 'polite'
        }
        
        self.negative_words = {
            'terrible', 'horrible', 'awful', 'dreadful', 'atrocious',
            'appalling', 'disgraceful', 'shocking', 'horrific', 'nightmare',
            'bad', 'poor', 'subpar', 'inadequate', 'unsatisfactory',
            'rude', 'dismissive', 'negligent', 'incompetent', 'unprofessional',
            'rushed', 'ignored', 'careless', 'unhelpful', 'cold',
            'arrogant', 'impatient', 'neglectful', 'unavailable', 'ineffective'
        }
        
        # Neutral words
        self.neutral_words = {
            'okay', 'average', 'normal', 'standard', 'regular', 'typical',
            'mediocre', 'moderate', 'passable', 'tolerable', 'alright',
            'so-so', 'fine', 'decent', 'acceptable', 'reasonable', 'adequate',
            'ordinary', 'common', 'usual', 'routine', 'basic', 'simple',
            'fair', 'moderate', 'reasonable'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'absolutely': 2.0, 'completely': 1.8,
            'totally': 1.8, 'really': 1.5, 'quite': 1.4, 'highly': 1.6,
            'exceptionally': 2.0, 'remarkably': 1.7, 'incredibly': 1.9
        }
        
        self.diminishers = {
            'slightly': 0.6, 'somewhat': 0.7, 'a bit': 0.6, 'a little': 0.6,
            'fairly': 0.8, 'relatively': 0.8, 'moderately': 0.7, 'partially': 0.7
        }
        
        # Negations
        self.negations = {'not', 'no', 'never', 'none', 'neither', 'nor',
                         "don't", "doesn't", "didn't", "won't", "wouldn't",
                         "couldn't", "shouldn't", "isn't", "aren't", "wasn't"}
        
        # Contrast words
        self.contrast_words = {'but', 'however', 'although', 'though', 'yet',
                              'except', 'nevertheless', 'nonetheless'}
    
    def preprocess(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def is_neutral_phrase(self, text):
        """Check if text contains neutral phrases (HIGHEST PRIORITY)"""
        text_lower = text.lower()
        
        for phrase in self.neutral_phrases:
            if phrase in text_lower:
                return True
        
        if re.search(r'nothing\s+(special|great|bad|good|terrible|horrible|exceptional)', text_lower):
            return True
        
        if re.search(r'neither\s+\w+\s+nor\s+\w+', text_lower):
            return True
        
        return False
    
    def has_strong_sentiment(self, text):
        """Check for strong sentiment that overrides neutral"""
        text_lower = text.lower()
        
        for word in self.strong_positive:
            if word in text_lower:
                return 'positive', 0.92
        
        for word in self.strong_negative:
            if word in text_lower:
                return 'negative', 0.92
        
        return None, 0.0
    
    def calculate_sentiment_score(self, text):
        """Calculate raw sentiment score (-1 to +1)"""
        words = text.split()
        
        total_score = 0.0
        word_count = 0
        
        negation_active = False
        negation_window = 0
        
        for i, word in enumerate(words):
            if word in self.neutral_indicators:
                continue
            
            if word in self.negations:
                negation_active = True
                negation_window = 3
                continue
            
            modifier = 1.0
            if word in self.intensifiers:
                modifier = self.intensifiers[word]
                continue
            elif word in self.diminishers:
                modifier = self.diminishers[word]
                continue
            
            word_score = 0
            
            if word in self.positive_words:
                word_score = 1 * modifier
            elif word in self.negative_words:
                word_score = -1 * modifier
            elif word in self.neutral_words:
                word_score = 0
            
            if negation_active:
                word_score = -word_score
            
            total_score += word_score
            if word_score != 0:
                word_count += 1
            
            if negation_active:
                negation_window -= 1
                if negation_window <= 0:
                    negation_active = False
        
        if word_count > 0:
            max_possible = word_count * 2
            normalized_score = total_score / max_possible
            return max(-1.0, min(1.0, normalized_score))
        
        return 0.0
    
    def detect_mixed_sentiment(self, text):
        """Detect if review has mixed sentiment"""
        text_lower = text.lower()
        
        has_contrast = any(word in text_lower for word in self.contrast_words)
        
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        strong_pos_count = sum(1 for word in self.strong_positive if word in text_lower)
        strong_neg_count = sum(1 for word in self.strong_negative if word in text_lower)
        
        if has_contrast and (pos_count > 0 or strong_pos_count > 0) and (neg_count > 0 or strong_neg_count > 0):
            return True, 0.75
        if (pos_count >= 2 and neg_count >= 2) or (strong_pos_count >= 1 and strong_neg_count >= 1):
            return True, 0.72
        
        return False, 0.0
    
    def rule_based_predict(self, text):
        """Balanced rule-based prediction"""
        
        if self.is_neutral_phrase(text):
            return 'neutral', 0.85
        
        strong_sentiment, strong_conf = self.has_strong_sentiment(text)
        if strong_sentiment:
            return strong_sentiment, strong_conf
        
        score = self.calculate_sentiment_score(text)
        is_mixed, mixed_conf = self.detect_mixed_sentiment(text)
        
        if is_mixed:
            return 'neutral', mixed_conf
        
        if score > 0.2:
            confidence = min(0.70 + abs(score) * 0.25, 0.95)
            return 'positive', confidence
        elif score < -0.2:
            confidence = min(0.70 + abs(score) * 0.25, 0.95)
            return 'negative', confidence
        else:
            return 'neutral', 0.72
    
    def vader_predict(self, text):
        """VADER-based prediction"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if self.is_neutral_phrase(text):
            return 'neutral', 0.80
        
        if compound > 0.25:
            confidence = min(0.70 + abs(compound), 0.95)
            return 'positive', confidence
        elif compound < -0.25:
            confidence = min(0.70 + abs(compound), 0.95)
            return 'negative', confidence
        else:
            return 'neutral', 0.68
    
    def textblob_predict(self, text):
        """TextBlob-based prediction"""
        if self.is_neutral_phrase(text):
            return 'neutral', 0.82
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.15:
            confidence = min(0.65 + abs(polarity), 0.90)
            return 'positive', confidence
        elif polarity < -0.15:
            confidence = min(0.65 + abs(polarity), 0.90)
            return 'negative', confidence
        else:
            return 'neutral', 0.68
    
    def ml_predict(self, text):
        """ML model prediction"""
        if self.ml_model is None or self.vectorizer is None:
            return None, 0.0
        
        try:
            X = self.vectorizer.transform([text])
            proba = self.ml_model.predict_proba(X)[0]
            idx = np.argmax(proba)
            prediction = self.ml_model.classes_[idx]
            confidence = proba[idx]
            return prediction, confidence
        except:
            return None, 0.0
    
    def predict(self, text):
        """Master prediction"""
        if not text or len(text.strip()) < 2:
            return "Neutral (Invalid Input - Confidence: 55%)", 0.55
        
        text = self.preprocess(text)
        word_count = len(text.split())
        is_mixed, _ = self.detect_mixed_sentiment(text)
        
        rule_sentiment, rule_conf = self.rule_based_predict(text)
        vader_sentiment, vader_conf = self.vader_predict(text)
        blob_sentiment, blob_conf = self.textblob_predict(text)
        ml_sentiment, ml_conf = self.ml_predict(text)
        
        if is_mixed:
            weights = {'rule': 0.50, 'vader': 0.25, 'blob': 0.15, 'ml': 0.10}
        elif word_count < 5:
            weights = {'rule': 0.35, 'vader': 0.40, 'blob': 0.15, 'ml': 0.10}
        elif word_count > 50:
            weights = {'rule': 0.25, 'vader': 0.20, 'blob': 0.15, 'ml': 0.40}
        else:
            weights = {'rule': 0.35, 'vader': 0.30, 'blob': 0.15, 'ml': 0.20}
        
        scores = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        scores[rule_sentiment] += rule_conf * weights['rule']
        scores[vader_sentiment] += vader_conf * weights['vader']
        scores[blob_sentiment] += blob_conf * weights['blob']
        
        if ml_sentiment:
            scores[ml_sentiment] += ml_conf * weights['ml']
        
        total_score = sum(scores.values())
        
        if total_score == 0:
            final_sentiment = 'neutral'
            final_confidence = 0.65
        else:
            final_sentiment = max(scores, key=scores.get)
            final_confidence = scores[final_sentiment] / total_score
            final_confidence = min(max(final_confidence, 0.65), 0.95)
        
        text_lower = text.lower()
        
        if 'nothing special' in text_lower:
            final_sentiment = 'neutral'
            final_confidence = 0.85
        elif 'nothing great' in text_lower or 'nothing bad' in text_lower:
            final_sentiment = 'neutral'
            final_confidence = 0.82
        
        sentiment_display = final_sentiment.capitalize()
        
        return f"{sentiment_display} (Healthcare AI - Confidence: {final_confidence:.1%})", final_confidence


# ============================================
# SIMPLE FACTORY FUNCTIONS
# ============================================

def create_analyzer(ml_model=None, vectorizer=None):
    """
    Create a balanced analyzer instance
    Usage: analyzer = create_analyzer(ensemble, vectorizer)
    """
    return BalancedHealthcareAnalyzer(ml_model, vectorizer)


def create_analyzer_from_path(model_path=None, vectorizer_path=None):
    """
    Create analyzer by loading model from file paths
    Usage: analyzer = create_analyzer_from_path('models/model.pkl')
    """
    ml_model = None
    vectorizer = None
    
    if model_path and os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            if 'ensemble' in model_data:
                ml_model = model_data['ensemble']
            elif 'model' in model_data:
                ml_model = model_data['model']
            else:
                ml_model = model_data
            
            if vectorizer_path and os.path.exists(vectorizer_path):
                vectorizer = joblib.load(vectorizer_path)
            elif 'vectorizer' in model_data:
                vectorizer = model_data['vectorizer']
        except Exception as e:
            print(f"Error loading model: {e}")
    
    return BalancedHealthcareAnalyzer(ml_model, vectorizer)