"""
Aspect-Based Sentiment Analysis for Healthcare Feedback
Analyzes different aspects of patient feedback separately
"""

import re
from typing import Dict, List, Tuple, Optional

class AspectBasedAnalyzer:
    """Analyze different aspects of healthcare feedback separately"""
    
    def __init__(self):
        # Define aspects with their keywords
        self.aspects = {
            'doctor': {
                'keywords': ['doctor', 'physician', 'dr.', 'specialist', 'surgeon', 'medical staff'],
                'positive': [
                    'compassionate', 'thorough', 'explained', 'listened', 'knowledgeable', 
                    'caring', 'attentive', 'professional', 'excellent', 'amazing',
                    'understood', 'answered', 'clear', 'helpful', 'patient'
                ],
                'negative': [
                    'rushed', 'dismissive', 'arrogant', 'didn\'t listen', 'ignored',
                    'unprofessional', 'incompetent', 'rude', 'hurried', 'careless',
                    'wrong', 'misdiagnosed', 'negligent'
                ]
            },
            'nurse': {
                'keywords': ['nurse', 'nursing', 'staff', 'attendant', 'caregiver'],
                'positive': [
                    'attentive', 'caring', 'helpful', 'compassionate', 'responsive',
                    'kind', 'gentle', 'supportive', 'wonderful', 'dedicated'
                ],
                'negative': [
                    'rude', 'ignored', 'slow', 'unprofessional', 'neglectful',
                    'disrespectful', 'lazy', 'inattentive', 'busy', 'unhelpful'
                ]
            },
            'facility': {
                'keywords': ['facility', 'room', 'hospital', 'clinic', 'building', 'environment', 'floor'],
                'positive': [
                    'clean', 'well-maintained', 'comfortable', 'modern', 'organized',
                    'spacious', 'quiet', 'nice', 'beautiful', 'new'
                ],
                'negative': [
                    'dirty', 'unclean', 'messy', 'cramped', 'outdated', 'old',
                    'noisy', 'crowded', 'broken', 'filthy'
                ]
            },
            'wait_time': {
                'keywords': ['wait', 'time', 'delay', 'hours', 'minute', 'appointment', 'scheduling'],
                'positive': [
                    'short', 'quick', 'fast', 'no wait', 'prompt', 'on time',
                    'efficient', 'immediate', 'minimal'
                ],
                'negative': [
                    'long', 'excessive', 'hours', 'delayed', 'extended',
                    'forever', 'endless', 'late', 'behind'
                ]
            },
            'billing': {
                'keywords': ['billing', 'cost', 'price', 'insurance', 'payment', 'expensive', 'affordable'],
                'positive': [
                    'reasonable', 'affordable', 'covered', 'clear', 'transparent',
                    'fair', 'helpful', 'explained'
                ],
                'negative': [
                    'expensive', 'overcharged', 'wrong', 'mistake', 'confusing',
                    'unreasonable', 'high', 'costly', 'denied'
                ]
            }
        }
        
        # Contrast words that indicate mixed sentiment
        self.contrast_words = ['but', 'however', 'although', 'though', 'yet', 'except']
        
        # Neutral indicators
        self.neutral_indicators = ['okay', 'average', 'decent', 'fine', 'not bad', 'could be better']
    
    def extract_aspects(self, text: str) -> Dict[str, Dict]:
        """
        Extract sentiment for each aspect mentioned in the text
        
        Returns:
            Dict with aspect name as key and sentiment info as value
        """
        text_lower = text.lower()
        results = {}
        
        for aspect, config in self.aspects.items():
            # Check if this aspect is mentioned
            mentioned = any(keyword in text_lower for keyword in config['keywords'])
            
            if mentioned:
                # Count positive and negative indicators for this aspect
                pos_count = sum(1 for word in config['positive'] if word in text_lower)
                neg_count = sum(1 for word in config['negative'] if word in text_lower)
                
                # Determine sentiment for this aspect
                if pos_count > neg_count:
                    sentiment = 'positive'
                    confidence = min(0.70 + (pos_count * 0.05), 0.95)
                elif neg_count > pos_count:
                    sentiment = 'negative'
                    confidence = min(0.70 + (neg_count * 0.05), 0.95)
                else:
                    sentiment = 'neutral'
                    confidence = 0.65
                
                results[aspect] = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'positive_count': pos_count,
                    'negative_count': neg_count,
                    'mentioned': True
                }
        
        return results
    
    def detect_mixed_sentiment(self, text: str, aspect_results: Dict) -> Tuple[bool, str]:
        """
        Detect if the feedback has mixed sentiments across different aspects
        """
        if len(aspect_results) < 2:
            return False, "single_aspect"
        
        sentiments = [info['sentiment'] for info in aspect_results.values()]
        
        # Check if we have both positive and negative aspects
        has_positive = 'positive' in sentiments
        has_negative = 'negative' in sentiments
        
        # Check for contrast words
        text_lower = text.lower()
        has_contrast = any(word in text_lower for word in self.contrast_words)
        
        if has_positive and has_negative:
            return True, "mixed_aspects"
        elif has_contrast:
            return True, "contrast_words"
        
        return False, "unified"
    
    def calculate_overall_sentiment(self, aspect_results: Dict, text: str) -> Tuple[str, float, Dict]:
        """
        Calculate overall sentiment based on aspect analysis
        """
        if not aspect_results:
            return None, 0.0, {}
        
        # Detect mixed sentiment
        is_mixed, mixed_type = self.detect_mixed_sentiment(text, aspect_results)
        
        if is_mixed:
            # This is a mixed review - return neutral
            return 'neutral', 0.75, aspect_results
        
        # Count positive and negative aspects
        pos_aspects = sum(1 for info in aspect_results.values() if info['sentiment'] == 'positive')
        neg_aspects = sum(1 for info in aspect_results.values() if info['sentiment'] == 'negative')
        neu_aspects = sum(1 for info in aspect_results.values() if info['sentiment'] == 'neutral')
        
        total_aspects = len(aspect_results)
        
        if pos_aspects > neg_aspects:
            # More positive aspects
            confidence = 0.70 + (pos_aspects / total_aspects) * 0.20
            return 'positive', min(confidence, 0.95), aspect_results
        elif neg_aspects > pos_aspects:
            # More negative aspects
            confidence = 0.70 + (neg_aspects / total_aspects) * 0.20
            return 'negative', min(confidence, 0.95), aspect_results
        else:
            # Equal positive and negative
            return 'neutral', 0.70, aspect_results
    
    def analyze(self, text: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        Main method to analyze text
        
        Returns:
            Tuple of (overall_sentiment, aspect_results, summary_text)
        """
        # Extract aspect sentiments
        aspect_results = self.extract_aspects(text)
        
        # Calculate overall sentiment
        overall, confidence, aspects = self.calculate_overall_sentiment(aspect_results, text)
        
        # Generate summary
        summary = self.generate_summary(aspect_results, overall, confidence)
        
        return overall, aspects, summary
    
    def generate_summary(self, aspect_results: Dict, overall: str, confidence: float) -> str:
        """
        Generate a human-readable summary of the analysis
        """
        if not aspect_results:
            return "No specific aspects identified in this feedback."
        
        summary_parts = []
        
        for aspect, info in aspect_results.items():
            sentiment = info['sentiment']
            if sentiment == 'positive':
                emoji = "✅"
                text = "positive"
            elif sentiment == 'negative':
                emoji = "❌"
                text = "negative"
            else:
                emoji = "➖"
                text = "neutral"
            
            aspect_name = aspect.replace('_', ' ').title()
            summary_parts.append(f"{emoji} {aspect_name}: {text}")
        
        summary = " | ".join(summary_parts)
        
        if overall:
            overall_text = "positive" if overall == 'positive' else "negative" if overall == 'negative' else "mixed/neutral"
            summary = f"Overall: {overall_text} ({confidence:.0%} confidence) | {summary}"
        
        return summary
    
    def get_actionable_insights(self, aspect_results: Dict) -> List[str]:
        """
        Generate actionable insights based on aspect analysis
        """
        insights = []
        
        for aspect, info in aspect_results.items():
            if info['sentiment'] == 'negative':
                if aspect == 'doctor':
                    insights.append("📋 Schedule physician communication training")
                elif aspect == 'nurse':
                    insights.append("👩‍⚕️ Review nursing staff responsiveness protocols")
                elif aspect == 'wait_time':
                    insights.append("⏰ Optimize patient scheduling and reduce wait times")
                elif aspect == 'facility':
                    insights.append("🏥 Conduct facility maintenance and cleanliness audit")
                elif aspect == 'billing':
                    insights.append("💰 Review billing processes and patient communication")
        
        return insights


# Standalone function for easy integration
def analyze_healthcare_feedback(text: str) -> Dict:
    """
    Easy-to-use function for analyzing healthcare feedback
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = AspectBasedAnalyzer()
    overall, aspects, summary = analyzer.analyze(text)
    insights = analyzer.get_actionable_insights(aspects if aspects else {})
    
    return {
        'sentiment': overall if overall else 'neutral',
        'aspects': aspects if aspects else {},
        'summary': summary,
        'insights': insights,
        'has_mixed_sentiment': overall == 'neutral' and aspects and len(aspects) > 1
    }


# Test the analyzer
if __name__ == "__main__":
    # Test cases
    test_reviews = [
        "The facility was clean and well-maintained, but the wait time was excessive. The doctor was okay but rushed through my appointment.",
        "The doctor was very compassionate and explained everything clearly.",
        "The nurse ignored my calls for help and was very rude.",
        "Great doctor, terrible wait time, but clean facility.",
        "The staff was friendly and the facility was clean. Excellent experience!",
        "The billing department made a mistake and the wait was too long."
    ]
    
    print("\n" + "="*70)
    print("ASPECT-BASED SENTIMENT ANALYSIS - TEST RESULTS")
    print("="*70)
    
    for review in test_reviews:
        result = analyze_healthcare_feedback(review)
        
        print(f"\n📝 Review: {review[:80]}...")
        print(f"   Overall Sentiment: {result['sentiment'].upper()}")
        print(f"   Summary: {result['summary']}")
        
        if result['insights']:
            print(f"   Actionable Insights:")
            for insight in result['insights']:
                print(f"      {insight}")
        
        if result['has_mixed_sentiment']:
            print(f"   ⚠️ Mixed Sentiment Detected")
        
        print("-"*50)