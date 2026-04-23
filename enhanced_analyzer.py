"""
enhanced_analyzer.py
MULTI-METHOD VOTING SYSTEM for Balanced Healthcare Sentiment Analysis
"""

import re
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

class EnhancedHealthcareAnalyzer:
    """
    Multi-method voting sentiment analyzer
    Combines: Rule-Based + VADER + TextBlob + Pattern Matching
    """
    
    def __init__(self, ml_model=None, vectorizer=None):
        self.sia = SentimentIntensityAnalyzer()
        self.ml_model = ml_model
        self.vectorizer = vectorizer
        
        # ============================================
        # COMPREHENSIVE LEXICONS
        # ============================================
        
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'perfect', 'exceptional', 'incredible', 'awesome', 'brilliant',
            'life-saving', 'saved my life', 'great', 'good', 'nice', 'pleasant',
            'satisfactory', 'compassionate', 'caring', 'attentive', 'professional',
            'thorough', 'helpful', 'friendly', 'comfortable', 'efficient',
            'recovered', 'grateful', 'satisfied', 'healed', 'improved'
        }
        
        self.negative_words = {
            'terrible', 'horrible', 'awful', 'dreadful', 'atrocious',
            'appalling', 'disgraceful', 'shocking', 'horrific', 'nightmare',
            'bad', 'poor', 'subpar', 'inadequate', 'unsatisfactory',
            'rude', 'dismissive', 'negligent', 'incompetent', 'unprofessional',
            'rushed', 'ignored', 'careless', 'unhelpful', 'cold',
            'arrogant', 'impatient', 'neglectful', 'ineffective', 'waste'
        }
        
        self.neutral_words = {
            'okay', 'fine', 'average', 'normal', 'standard', 'regular',
            'typical', 'ordinary', 'mediocre', 'moderate', 'passable',
            'tolerable', 'alright', 'decent', 'acceptable', 'adequate',
            'fair', 'so-so', 'nothing special', 'nothing great', 'nothing bad'
        }
        
        # ============================================
        # NEUTRAL PHRASES (High priority)
        # ============================================
        
        self.neutral_phrases = [
            'nothing special', 'nothing great', 'nothing bad', 'nothing terrible',
            'neither good nor bad', 'not good not bad', 'could be better could be worse',
            'no strong feelings', 'no complaints no praise', 'mixed feelings',
            'just okay', 'just fine', 'just average', 'just normal',
            'doctor was okay', 'doctor is okay', 'okay doctor', 'okay experience',
            'average experience', 'normal experience', 'decent experience',
            'so-so experience', 'fine but nothing special', 'worked but slowly',
            'nothing very good or very bad', 'treatment is acceptable',
            'the service is okay', 'it was okay', 'fairly normal', 'nothing exceptional'
        ]
        
        # ============================================
        # MIXED SENTIMENT PATTERNS
        # ============================================
        
        self.mixed_patterns = [
            r'good\s+but\s+(bad|terrible|awful|poor)',
            r'great\s+but\s+(bad|terrible|awful|poor)',
            r'excellent\s+but\s+(bad|terrible|awful|poor)',
            r'nice\s+but\s+(bad|terrible|awful|poor)',
            r'clean\s+but\s+(dirty|unclean)',
            r'caring\s+but\s+(rude|dismissive)',
            r'professional\s+but\s+(rude|unprofessional)',
            r'but.*wait.*long', r'but.*expensive', r'but.*slow',
            r'however.*bad', r'however.*poor', r'although.*bad'
        ]
        
        # ============================================
        # STRONG INDICATORS (Override)
        # ============================================
        
        self.strong_positive = {
            'excellent', 'amazing', 'outstanding', 'perfect', 'life-saving',
            'saved my life', 'best ever', 'could not be better'
        }
        
        self.strong_negative = {
            'terrible', 'horrible', 'awful', 'worst', 'never again',
            'complete waste', 'absolutely terrible', 'nightmare'
        }
        
        # ============================================
        # NEGATION PATTERNS
        # ============================================
        
        self.negations = {'not', 'no', 'never', 'none', "don't", "doesn't",
                         "didn't", "won't", "wouldn't", "couldn't", "shouldn't"}
        
        # ============================================
        # METHOD WEIGHTS (Can be adjusted)
        # ============================================
        
        self.weights = {
            'rule_based': 0.35,
            'vader': 0.30,
            'textblob': 0.20,
            'pattern': 0.15
        }
    
    def preprocess(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # ============================================
    # METHOD 1: RULE-BASED (Word counting)
    # ============================================
    
    def rule_based_predict(self, text):
        """Count positive and negative words"""
        text_lower = text.lower()
        
        pos_count = sum(1 for w in self.positive_words if w in text_lower)
        neg_count = sum(1 for w in self.negative_words if w in text_lower)
        neu_count = sum(1 for w in self.neutral_words if w in text_lower)
        
        # Check for neutral phrases first
        for phrase in self.neutral_phrases:
            if phrase in text_lower:
                return 'neutral', 0.85
        
        # Check for mixed sentiment
        for pattern in self.mixed_patterns:
            if re.search(pattern, text_lower):
                return 'neutral', 0.80
        
        if pos_count > neg_count and pos_count > neu_count:
            confidence = min(0.70 + (pos_count * 0.05), 0.90)
            return 'positive', confidence
        elif neg_count > pos_count and neg_count > neu_count:
            confidence = min(0.70 + (neg_count * 0.05), 0.90)
            return 'negative', confidence
        elif neu_count >= pos_count and neu_count >= neg_count and neu_count > 0:
            return 'neutral', 0.75
        else:
            return 'neutral', 0.65
    
    # ============================================
    # METHOD 2: VADER
    # ============================================
    
    def vader_predict(self, text):
        """VADER sentiment analysis"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound > 0.3:
            confidence = min(0.70 + compound * 0.2, 0.92)
            return 'positive', confidence
        elif compound < -0.3:
            confidence = min(0.70 + abs(compound) * 0.2, 0.92)
            return 'negative', confidence
        else:
            return 'neutral', 0.72
    
    # ============================================
    # METHOD 3: TEXTBLOB
    # ============================================
    
    def textblob_predict(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.2:
            confidence = min(0.65 + polarity * 0.2, 0.88)
            return 'positive', confidence
        elif polarity < -0.2:
            confidence = min(0.65 + abs(polarity) * 0.2, 0.88)
            return 'negative', confidence
        else:
            return 'neutral', 0.70
    
    # ============================================
    # METHOD 4: PATTERN MATCHING
    # ============================================
    
    def pattern_predict(self, text):
        """Pattern-based prediction"""
        text_lower = text.lower()
        
        # Check strong positives
        for word in self.strong_positive:
            if word in text_lower:
                return 'positive', 0.92
        
        # Check strong negatives
        for word in self.strong_negative:
            if word in text_lower:
                return 'negative', 0.92
        
        # Check neutral phrases
        for phrase in self.neutral_phrases:
            if phrase in text_lower:
                return 'neutral', 0.88
        
        # Check mixed patterns
        for pattern in self.mixed_patterns:
            if re.search(pattern, text_lower):
                return 'neutral', 0.82
        
        # Check negation patterns
        if re.search(r'not\s+(good|great|nice|clean|friendly|helpful)', text_lower):
            return 'negative', 0.85
        if re.search(r'not\s+(bad|terrible|awful|horrible)', text_lower):
            return 'positive', 0.85
        
        return None, 0.0
    
    # ============================================
    # VOTING SYSTEM
    # ============================================
    
    def weighted_vote(self, predictions):
        """
        Weighted voting to determine final sentiment
        predictions: list of (sentiment, confidence, method_name)
        """
        scores = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        for sentiment, confidence, method_name in predictions:
            if sentiment:
                weight = self.weights.get(method_name, 0.25)
                scores[sentiment] += confidence * weight
        
        # Find sentiment with highest score
        if max(scores.values()) == 0:
            return 'neutral', 0.65
        
        best_sentiment = max(scores, key=scores.get)
        best_score = scores[best_sentiment]
        total_score = sum(scores.values())
        
        if total_score > 0:
            confidence = best_score / total_score
            confidence = min(max(confidence, 0.65), 0.92)
        else:
            confidence = 0.70
        
        return best_sentiment, confidence
    
    # ============================================
    # MAIN PREDICT METHOD
    # ============================================
    
    def predict(self, text):
        """
        Multi-method voting prediction
        ALL methods vote, weighted voting decides final sentiment
        """
        
        if not text or len(text.strip()) < 2:
            return "Neutral (Invalid Input - Confidence: 55%)", 0.55
        
        text = self.preprocess(text)
        
        # Collect predictions from ALL methods
        predictions = []
        
        # Method 1: Rule-Based
        rule_sentiment, rule_conf = self.rule_based_predict(text)
        predictions.append((rule_sentiment, rule_conf, 'rule_based'))
        
        # Method 2: VADER
        vader_sentiment, vader_conf = self.vader_predict(text)
        predictions.append((vader_sentiment, vader_conf, 'vader'))
        
        # Method 3: TextBlob
        blob_sentiment, blob_conf = self.textblob_predict(text)
        predictions.append((blob_sentiment, blob_conf, 'textblob'))
        
        # Method 4: Pattern Matching
        pattern_sentiment, pattern_conf = self.pattern_predict(text)
        if pattern_sentiment:
            predictions.append((pattern_sentiment, pattern_conf, 'pattern'))
        
        # Weighted voting
        final_sentiment, final_confidence = self.weighted_vote(predictions)
        
        # Special override for "okay" - force neutral if no strong signals
        text_lower = text.lower()
        if 'okay' in text_lower and 'very good' not in text_lower and 'excellent' not in text_lower:
            if final_sentiment != 'negative':
                final_sentiment = 'neutral'
                final_confidence = max(final_confidence, 0.78)
        
        # Special override for mixed sentiment with "but"
        if ' but ' in text_lower:
            pos_count = sum(1 for w in self.positive_words if w in text_lower)
            neg_count = sum(1 for w in self.negative_words if w in text_lower)
            if pos_count >= 1 and neg_count >= 1:
                final_sentiment = 'neutral'
                final_confidence = max(final_confidence, 0.80)
        
        sentiment_display = final_sentiment.capitalize()
        
        return f"{sentiment_display} (Healthcare AI - Confidence: {final_confidence:.1%})", final_confidence


# ============================================
# FACTORY FUNCTION
# ============================================

def create_enhanced_analyzer(ml_model=None, vectorizer=None):
    """Create analyzer instance"""
    return EnhancedHealthcareAnalyzer(ml_model, vectorizer)