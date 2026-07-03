#!/usr/bin/env python3
"""
Test script to verify the mood analyzer implementation.
Shows tokens, scores, and predictions for each sample post.
"""

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

def test_preprocess():
    """Test the preprocess method with examples."""
    print("=" * 70)
    print("TESTING PREPROCESS METHOD")
    print("=" * 70)
    
    analyzer = MoodAnalyzer()
    
    # Test a few posts to show tokenization
    test_posts = [
        "I love this class so much",
        "This is so boring 😒",
        "I am not happy about this",
    ]
    
    for post in test_posts:
        tokens = analyzer.preprocess(post)
        print(f"\nOriginal: {post}")
        print(f"Tokens:   {tokens}")

def test_score_text():
    """Test the score_text method."""
    print("\n" + "=" * 70)
    print("TESTING SCORE_TEXT METHOD")
    print("=" * 70)
    
    analyzer = MoodAnalyzer()
    
    test_posts = [
        "I love this class so much",
        "Today was a terrible day",
        "I am not happy about this",
        "Feeling tired but kind of hopeful",
    ]
    
    for post in test_posts:
        tokens = analyzer.preprocess(post)
        score = analyzer.score_text(post)
        print(f"\nPost:  {post}")
        print(f"Tokens: {tokens}")
        print(f"Score:  {score}")

def test_predict_label():
    """Test the predict_label method."""
    print("\n" + "=" * 70)
    print("TESTING PREDICT_LABEL METHOD")
    print("=" * 70)
    
    analyzer = MoodAnalyzer()
    
    test_posts = [
        "I love this class so much",
        "Today was a terrible day",
        "I am not happy about this",
        "Feeling tired but kind of hopeful",
        "This is fine",
    ]
    
    for post in test_posts:
        label = analyzer.predict_label(post)
        score = analyzer.score_text(post)
        print(f"\nPost:  {post}")
        print(f"Score: {score}")
        print(f"Label: {label}")

def test_all_samples():
    """Test on the full dataset and show accuracy."""
    print("\n" + "=" * 70)
    print("TESTING ON FULL DATASET")
    print("=" * 70)
    
    analyzer = MoodAnalyzer()
    
    correct = 0
    for i, (post, true_label) in enumerate(zip(SAMPLE_POSTS, TRUE_LABELS)):
        predicted_label = analyzer.predict_label(post)
        score = analyzer.score_text(post)
        is_correct = predicted_label == true_label
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Post {i+1}: {post}")
        print(f"  Score: {score}, Predicted: {predicted_label}, True: {true_label}")
    
    accuracy = correct / len(SAMPLE_POSTS) * 100
    print(f"\n" + "=" * 70)
    print(f"ACCURACY: {correct}/{len(SAMPLE_POSTS)} ({accuracy:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    test_preprocess()
    test_score_text()
    test_predict_label()
    test_all_samples()
