from enhanced_analyzer import create_enhanced_analyzer

analyzer = create_enhanced_analyzer()

test_reviews = [
    # Positive
    ("Doctor treatment is very good and I am very happy", "positive"),
    ("The doctor is excellent and very caring", "positive"),
    ("Very good service and professional doctor", "positive"),
    
    # Negative
    ("Doctor treatment is very bad and I am very unhappy", "negative"),
    ("The treatment did not work and was very poor", "negative"),
    ("Waste of money and time", "negative"),
    
    # Neutral
    ("Doctor treatment is okay and normal", "neutral"),
    ("The experience was average and acceptable", "neutral"),
    ("Nothing very good or very bad", "neutral"),
    ("The treatment is acceptable", "neutral"),
    ("Doctor is fine but nothing special", "neutral"),
]

print("\n" + "="*70)
print("MULTI-METHOD VOTING SYSTEM TEST")
print("="*70)

correct = 0
for review, expected in test_reviews:
    result, conf = analyzer.predict(review)
    predicted = 'positive' if 'Positive' in result else 'negative' if 'Negative' in result else 'neutral'
    is_correct = (predicted == expected)
    correct += is_correct
    
    status = "✅" if is_correct else "❌"
    print(f"\n{status} Expected: {expected.upper():8} | Got: {predicted.upper():8} | Conf: {conf:.1%}")
    print(f"   Review: {review}")

print("\n" + "="*70)
print(f"RESULTS: {correct}/{len(test_reviews)} correct ({correct/len(test_reviews)*100:.1f}%)")
print("="*70)