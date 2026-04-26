"""
COMPREHENSIVE SYSTEM ACCURACY & PERFORMANCE TEST
Tests: Accuracy, Speed, Consistency, and Reliability
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import json

from ultimate_ensemble import create_ultimate_analyzer

class SystemTester:
    """Complete testing suite for healthcare sentiment system"""
    
    def __init__(self):
        print("🚀 Initializing System Tester...")
        self.analyzer = create_ultimate_analyzer()
        print("✅ Tester ready!\n")
        
        self.results = {
            'accuracy': {},
            'speed': {},
            'consistency': {},
            'errors': []
        }
    
    # ============================================
    # TEST 1: ACCURACY ON LABELED DATA
    # ============================================
    
    def test_accuracy(self, test_data):
        """Test accuracy on labeled dataset"""
        print("\n" + "="*70)
        print("📊 TEST 1: ACCURACY ON LABELED DATA")
        print("="*70)
        
        correct = 0
        total = len(test_data)
        results = []
        
        for review, expected in test_data:
            result, confidence = self.analyzer.predict(review)
            predicted = result.lower()
            
            is_correct = (predicted == expected)
            if is_correct:
                correct += 1
            
            results.append({
                'review': review[:60],
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Print progress
            if len(results) % 20 == 0:
                print(f"   Processed {len(results)}/{total} reviews...")
        
        accuracy = (correct / total) * 100
        
        print(f"\n✅ ACCURACY: {accuracy:.1f}% ({correct}/{total})")
        
        # Show incorrect predictions
        incorrect = [r for r in results if not r['correct']]
        if incorrect:
            print(f"\n❌ INCORRECT PREDICTIONS ({len(incorrect)}):")
            for i, err in enumerate(incorrect[:10], 1):
                print(f"   {i}. Expected: {err['expected'].upper()} | Got: {err['predicted'].upper()} | Conf: {err['confidence']:.1%}")
                print(f"      Review: {err['review']}...")
        
        self.results['accuracy'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'incorrect': len(incorrect),
            'details': results
        }
        
        return accuracy
    
    # ============================================
    # TEST 2: PERFORMANCE SPEED
    # ============================================
    
    def test_speed(self, reviews, iterations=3):
        """Test prediction speed"""
        print("\n" + "="*70)
        print("⚡ TEST 2: PERFORMANCE SPEED")
        print("="*70)
        
        speeds = []
        
        for i in range(iterations):
            start = time.time()
            for review in reviews:
                self.analyzer.predict(review)
            end = time.time()
            
            elapsed = end - start
            avg_per_review = (elapsed / len(reviews)) * 1000  # ms per review
            speeds.append(avg_per_review)
            
            print(f"   Run {i+1}: {elapsed:.3f}s total | {avg_per_review:.1f}ms per review")
        
        avg_speed = np.mean(speeds)
        min_speed = min(speeds)
        max_speed = max(speeds)
        
        print(f"\n📈 SPEED SUMMARY:")
        print(f"   Average: {avg_speed:.1f}ms per review")
        print(f"   Fastest: {min_speed:.1f}ms per review")
        print(f"   Slowest: {max_speed:.1f}ms per review")
        print(f"   Reviews per second: {1000/avg_speed:.1f}")
        
        self.results['speed'] = {
            'avg_ms_per_review': avg_speed,
            'min_ms': min_speed,
            'max_ms': max_speed,
            'reviews_per_second': 1000/avg_speed
        }
        
        return avg_speed
    
    # ============================================
    # TEST 3: CONSISTENCY (Same review, same result?)
    # ============================================
    
    def test_consistency(self, review, iterations=10):
        """Test if same review gets same prediction consistently"""
        print("\n" + "="*70)
        print("🔄 TEST 3: CONSISTENCY CHECK")
        print("="*70)
        
        predictions = []
        confidences = []
        
        for i in range(iterations):
            result, confidence = self.analyzer.predict(review)
            predictions.append(result)
            confidences.append(confidence)
        
        # Check consistency
        unique_predictions = set(predictions)
        is_consistent = len(unique_predictions) == 1
        
        print(f"\n📝 Review: {review[:80]}...")
        print(f"   Iterations: {iterations}")
        print(f"   Unique predictions: {unique_predictions}")
        print(f"   Consistent: {'✅ YES' if is_consistent else '❌ NO'}")
        
        if not is_consistent:
            print(f"   Prediction distribution: {Counter(predictions)}")
        
        print(f"   Confidence range: {min(confidences):.1%} - {max(confidences):.1%}")
        
        self.results['consistency'] = {
            'is_consistent': is_consistent,
            'unique_predictions': list(unique_predictions),
            'confidence_min': min(confidences),
            'confidence_max': max(confidences)
        }
        
        return is_consistent
    
    # ============================================
    # TEST 4: CONFIDENCE CALIBRATION
    # ============================================
    
    def test_confidence_calibration(self, test_data):
        """Check if confidence scores correlate with accuracy"""
        print("\n" + "="*70)
        print("📈 TEST 4: CONFIDENCE CALIBRATION")
        print("="*70)
        
        confidence_buckets = {
            '90-100%': {'correct': 0, 'total': 0},
            '80-89%': {'correct': 0, 'total': 0},
            '70-79%': {'correct': 0, 'total': 0},
            '60-69%': {'correct': 0, 'total': 0},
            '50-59%': {'correct': 0, 'total': 0},
            '<50%': {'correct': 0, 'total': 0}
        }
        
        for review, expected in test_data:
            result, confidence = self.analyzer.predict(review)
            predicted = result.lower()
            
            # Determine bucket
            if confidence >= 0.90:
                bucket = '90-100%'
            elif confidence >= 0.80:
                bucket = '80-89%'
            elif confidence >= 0.70:
                bucket = '70-79%'
            elif confidence >= 0.60:
                bucket = '60-69%'
            elif confidence >= 0.50:
                bucket = '50-59%'
            else:
                bucket = '<50%'
            
            confidence_buckets[bucket]['total'] += 1
            if predicted == expected:
                confidence_buckets[bucket]['correct'] += 1
        
        print("\n   Confidence | Correct/Total | Accuracy")
        print("   " + "-"*40)
        
        for bucket, data in confidence_buckets.items():
            if data['total'] > 0:
                acc = (data['correct'] / data['total']) * 100
                print(f"   {bucket:10} | {data['correct']:2}/{data['total']:2}         | {acc:5.1f}%")
        
        self.results['calibration'] = confidence_buckets
    
    # ============================================
    # TEST 5: BATCH PROCESSING SPEED
    # ============================================
    
    def test_batch_speed(self, reviews, batch_sizes=[10, 25, 50, 100]):
        """Test performance with different batch sizes"""
        print("\n" + "="*70)
        print("📦 TEST 5: BATCH PROCESSING SPEED")
        print("="*70)
        
        batch_results = []
        
        for batch_size in batch_sizes:
            if batch_size > len(reviews):
                continue
            
            # Take sample
            sample = reviews[:batch_size]
            
            # Measure sequential processing
            start_seq = time.time()
            for review in sample:
                self.analyzer.predict(review)
            seq_time = time.time() - start_seq
            
            # Measure pandas apply (batch)
            start_batch = time.time()
            import pandas as pd
            df = pd.DataFrame({'review': sample})
            df['result'] = df['review'].apply(lambda x: self.analyzer.predict(x)[0])
            batch_time = time.time() - start_batch
            
            print(f"\n   Batch size: {batch_size}")
            print(f"      Sequential: {seq_time:.3f}s ({seq_time/batch_size*1000:.1f}ms/ review)")
            print(f"      Pandas apply: {batch_time:.3f}s ({batch_time/batch_size*1000:.1f}ms/ review)")
            print(f"      Speedup: {seq_time/batch_time:.1f}x")
            
            batch_results.append({
                'batch_size': batch_size,
                'sequential_time': seq_time,
                'batch_time': batch_time,
                'speedup': seq_time / batch_time
            })
        
        self.results['batch_performance'] = batch_results
    
    # ============================================
    # TEST 6: ERROR ANALYSIS
    # ============================================
    
    def analyze_errors(self, test_data):
        """Analyze patterns in incorrect predictions"""
        print("\n" + "="*70)
        print("🔍 TEST 6: ERROR PATTERN ANALYSIS")
        print("="*70)
        
        errors = []
        
        for review, expected in test_data:
            result, confidence = self.analyzer.predict(review)
            predicted = result.lower()
            
            if predicted != expected:
                errors.append({
                    'review': review,
                    'expected': expected,
                    'predicted': predicted,
                    'confidence': confidence,
                    'length': len(review.split())
                })
        
        if not errors:
            print("\n✅ NO ERRORS FOUND! Perfect accuracy on test set!")
            return
        
        print(f"\n📊 Found {len(errors)} errors")
        
        # Error by type
        error_types = Counter([f"{e['expected']}→{e['predicted']}" for e in errors])
        print(f"\n   Error type distribution:")
        for error_type, count in error_types.most_common():
            print(f"      {error_type}: {count}")
        
        # Error by review length
        short_errors = [e for e in errors if e['length'] < 5]
        medium_errors = [e for e in errors if 5 <= e['length'] < 15]
        long_errors = [e for e in errors if e['length'] >= 15]
        
        print(f"\n   Error by review length:")
        print(f"      Short (<5 words): {len(short_errors)}")
        print(f"      Medium (5-15 words): {len(medium_errors)}")
        print(f"      Long (>15 words): {len(long_errors)}")
        
        # Show examples
        print(f"\n   Example errors:")
        for i, err in enumerate(errors[:5], 1):
            print(f"      {i}. '{err['review'][:60]}...'")
            print(f"         Expected: {err['expected']} | Got: {err['predicted']} (Conf: {err['confidence']:.1%})")
        
        self.results['errors'] = errors
    
    # ============================================
    # GENERATE REPORT
    # ============================================
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📋 FINAL TEST REPORT")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        # Save JSON report
        filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {filename}")
        
        # Print summary
        print("\n📊 SUMMARY:")
        if 'accuracy' in self.results:
            print(f"   ✅ Accuracy: {self.results['accuracy']['accuracy']:.1f}%")
        if 'speed' in self.results:
            print(f"   ⚡ Speed: {self.results['speed']['avg_ms_per_review']:.1f}ms per review")
        if 'consistency' in self.results:
            print(f"   🔄 Consistency: {'✅ Pass' if self.results['consistency']['is_consistent'] else '❌ Fail'}")
        
        return report


# ============================================
# LOAD TEST DATA
# ============================================

def load_test_data():
    """Create labeled test dataset"""
    
    test_data = [
        # POSITIVE REVIEWS (20)
        ("The doctor was absolutely amazing and saved my life", "positive"),
        ("Excellent care from the entire medical team", "positive"),
        ("The nurses were wonderful and very attentive", "positive"),
        ("Very happy with my treatment and recovery", "positive"),
        ("The doctor explained everything clearly and patiently", "positive"),
        ("I am completely satisfied with the care I received", "positive"),
        ("The staff went above and beyond for my family", "positive"),
        ("The facility was spotless and modern", "positive"),
        ("The surgeon was highly skilled and professional", "positive"),
        ("The medication worked perfectly with no side effects", "positive"),
        
        # NEGATIVE REVIEWS (20)
        ("The nurse ignored my calls for help for hours", "negative"),
        ("The doctor was rude and dismissed my concerns", "negative"),
        ("Worst hospital experience of my life", "negative"),
        ("The staff was unprofessional and overwhelmed", "negative"),
        ("The wait time was excessive and frustrating", "negative"),
        ("The facility was dirty and poorly maintained", "negative"),
        ("The doctor prescribed the wrong medication", "negative"),
        ("The surgeon botched my procedure", "negative"),
        ("The billing department made multiple errors", "negative"),
        ("The nurse was negligent and caused more pain", "negative"),
        
        # NEUTRAL REVIEWS (10)
        ("The doctor was okay, nothing special", "neutral"),
        ("Average experience overall", "neutral"),
        ("The facility is fine for what it is", "neutral"),
        ("The service was acceptable but not exceptional", "neutral"),
        ("Nothing special about this hospital", "neutral"),
        ("The care was adequate for my needs", "neutral"),
        ("The doctor did his job, nothing extra", "neutral"),
        ("The experience met my expectations", "neutral"),
        ("Neither good nor bad experience", "neutral"),
        ("The staff performed their duties as expected", "neutral"),
    ]
    
    return test_data


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 HEALTHCARE SENTIMENT SYSTEM - COMPLETE TEST")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize tester
    tester = SystemTester()
    
    # Load test data
    test_data = load_test_data()
    print(f"\n📊 Loaded {len(test_data)} labeled test cases")
    print(f"   Positive: {len([t for t in test_data if t[1]=='positive'])}")
    print(f"   Negative: {len([t for t in test_data if t[1]=='negative'])}")
    print(f"   Neutral: {len([t for t in test_data if t[1]=='neutral'])}")
    
    # Run tests
    tester.test_accuracy(test_data)
    tester.test_speed([t[0] for t in test_data], iterations=3)
    tester.test_consistency("The doctor was very good", iterations=10)
    tester.test_confidence_calibration(test_data)
    tester.test_batch_speed([t[0] for t in test_data])
    tester.analyze_errors(test_data)
    
    # Generate report
    tester.generate_report()
    
    print("\n" + "="*70)
    print(f"✅ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)