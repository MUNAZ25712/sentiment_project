"""
IMPROVE ML MODEL TRAINING
Creates better training data and retrains the model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

class ImprovedHealthcareModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        
        # Better ensemble with optimized hyperparameters
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=500,        # More trees
                    max_depth=20,            # Deeper trees
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced', # Handle imbalanced data
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )),
                ('lr', LogisticRegression(
                    max_iter=2000,
                    C=1.5,
                    class_weight='balanced',
                    random_state=42
                ))
            ],
            voting='soft',
            weights=[2, 1, 1]
        )
    
    def create_enhanced_dataset(self):
        """Create balanced dataset with more neutral examples"""
        
        # POSITIVE REVIEWS (400 examples)
        positive_reviews = [
            # Strong positive
            ("The doctor was absolutely amazing and saved my life!", "positive"),
            ("Excellent care from the entire medical team!", "positive"),
            ("The nurses were wonderful and very attentive.", "positive"),
            ("Best hospital experience I've ever had.", "positive"),
            ("I'm completely satisfied with my treatment.", "positive"),
            
            # Moderate positive
            ("The doctor explained everything clearly.", "positive"),
            ("Very good service and professional staff.", "positive"),
            ("The treatment worked very well.", "positive"),
            ("I'm very happy with the care I received.", "positive"),
            ("The staff was friendly and helpful.", "positive"),
        ] * 40  # Multiply to get 400 examples
        
        # NEGATIVE REVIEWS (400 examples)
        negative_reviews = [
            # Strong negative
            ("The doctor was absolutely terrible and rude!", "negative"),
            ("Worst hospital experience ever!", "negative"),
            ("The nurse ignored my calls for help.", "negative"),
            ("I'm completely dissatisfied with the treatment.", "negative"),
            ("The staff was negligent and unprofessional.", "negative"),
            
            # Moderate negative
            ("The doctor was rude and didn't listen.", "negative"),
            ("Very disappointed with the service.", "negative"),
            ("The treatment was not effective.", "negative"),
            ("The wait time was excessive.", "negative"),
            ("The facility was dirty.", "negative"),
        ] * 40  # Multiply to get 400 examples
        
        # NEUTRAL REVIEWS (200 examples - MORE THAN BEFORE!)
        neutral_reviews = [
            # Mixed sentiment
            ("The doctor was good but the wait was long.", "neutral"),
            ("The facility was clean but the staff was unfriendly.", "neutral"),
            ("Good medical care but poor communication.", "neutral"),
            ("The treatment worked but it was expensive.", "neutral"),
            ("The nurses were caring but the room was dirty.", "neutral"),
            
            # Average/Okay
            ("The doctor was okay, nothing special.", "neutral"),
            ("Average experience overall.", "neutral"),
            ("The service was acceptable.", "neutral"),
            ("Nothing special, nothing terrible.", "neutral"),
            ("The treatment is acceptable.", "neutral"),
            
            # Neutral phrases
            ("The doctor is fine but nothing special.", "neutral"),
            ("It was a typical hospital visit.", "neutral"),
            ("The facility is decent.", "neutral"),
            ("The care was adequate.", "neutral"),
            ("Not great but not terrible.", "neutral"),
        ] * 14  # Multiply to get ~200 examples
        
        # Combine all
        all_reviews = positive_reviews + negative_reviews + neutral_reviews
        
        # Shuffle
        df = pd.DataFrame(all_reviews, columns=['review', 'sentiment'])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Enhanced Dataset Created:")
        print(f"   Total: {len(df)} reviews")
        print(f"   Positive: {len(df[df['sentiment']=='positive'])}")
        print(f"   Negative: {len(df[df['sentiment']=='negative'])}")
        print(f"   Neutral: {len(df[df['sentiment']=='neutral'])}")
        
        return df
    
    def extract_features(self, text):
        """Extract additional features for better accuracy"""
        text_lower = text.lower()
        
        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capitals_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }
        
        # Add sentiment indicators
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'happy', 'satisfied']
        negative_indicators = ['bad', 'terrible', 'awful', 'horrible', 'unhappy', 'dissatisfied']
        
        features['pos_indicator_count'] = sum(1 for w in positive_indicators if w in text_lower)
        features['neg_indicator_count'] = sum(1 for w in negative_indicators if w in text_lower)
        
        return features
    
    def train(self):
        """Train improved model"""
        
        print("\n" + "="*60)
        print("TRAINING IMPROVED HEALTHCARE MODEL")
        print("="*60)
        
        # Create enhanced dataset
        df = self.create_enhanced_dataset()
        
        # Extract TF-IDF features
        print("\n🔄 Extracting TF-IDF features...")
        X_tfidf = self.vectorizer.fit_transform(df['review'])
        
        # Extract additional features
        print("🔄 Extracting additional features...")
        feature_list = [self.extract_features(review) for review in df['review']]
        feature_df = pd.DataFrame(feature_list)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        X_features = csr_matrix(feature_df.values.astype(np.float32))
        X_combined = hstack([X_tfidf, X_features])
        
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train ensemble
        print("\n🏋️ Training ensemble model...")
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"   ✅ Accuracy: {accuracy:.2%}")
        print(f"\n📊 Detailed Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble, X_combined, y, cv=5)
        print(f"\n🎯 Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Save model
        model_data = {
            'ensemble': self.ensemble,
            'vectorizer': self.vectorizer,
            'accuracy': accuracy
        }
        
        joblib.dump(model_data, 'models/healthcare_model_improved.pkl')
        print("\n💾 Model saved as 'models/healthcare_model_improved.pkl'")
        
        return accuracy

if __name__ == "__main__":
    trainer = ImprovedHealthcareModel()
    trainer.train()