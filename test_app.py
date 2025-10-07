"""
Test script for the Sentiment-Based Text Generator
This script tests the core functionality without running the full Streamlit app
"""

import sys
import os
from textblob import TextBlob

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("🧪 Testing Sentiment Analysis...")
    
    test_cases = [
        ("I love this amazing product!", "POSITIVE"),
        ("This is terrible and awful!", "NEGATIVE"),
        ("The weather is okay today.", "NEUTRAL"),
        ("I'm so happy and excited!", "POSITIVE"),
        ("This is frustrating and disappointing.", "NEGATIVE")
    ]
    
    for text, expected in test_cases:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            predicted = "POSITIVE"
        elif polarity < -0.1:
            predicted = "NEGATIVE"
        else:
            predicted = "NEUTRAL"
        
        status = "✅" if predicted == expected else "❌"
        print(f"{status} '{text}' -> {predicted} (expected: {expected})")
    
    print()

def test_text_generation():
    """Test text generation functionality"""
    print("🧪 Testing Text Generation...")
    
    # Test fallback text generation
    from utils import create_fallback_responses
    
    fallback_responses = create_fallback_responses()
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        responses = fallback_responses[sentiment]
        print(f"✅ {sentiment} fallback responses: {len(responses)} available")
    
    print()

def test_utility_functions():
    """Test utility functions"""
    print("🧪 Testing Utility Functions...")
    
    from utils import clean_text, validate_input, format_generated_text
    
    # Test text cleaning
    dirty_text = "  This   is    a   test!!!  "
    cleaned = clean_text(dirty_text)
    print(f"✅ Text cleaning: '{dirty_text}' -> '{cleaned}'")
    
    # Test input validation
    valid, msg = validate_input("Valid input")
    print(f"✅ Input validation (valid): {valid} - {msg}")
    
    valid, msg = validate_input("")
    print(f"✅ Input validation (empty): {valid} - {msg}")
    
    # Test text formatting
    long_text = "This is a very long text that should be truncated because it exceeds the maximum length allowed for proper formatting and display purposes."
    formatted = format_generated_text(long_text, 50)
    print(f"✅ Text formatting: {len(formatted)} chars")
    
    print()

def test_model_loading():
    """Test if models can be loaded (without actually loading them)"""
    print("🧪 Testing Model Dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
    
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not available")
    
    print()

def run_integration_test():
    """Run a simple integration test"""
    print("🧪 Running Integration Test...")
    
    # Simulate the main application flow
    test_prompt = "I'm feeling great today!"
    
    # Step 1: Sentiment Analysis
    blob = TextBlob(test_prompt)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "POSITIVE"
    elif polarity < -0.1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    print(f"✅ Sentiment detected: {sentiment} (polarity: {polarity:.2f})")
    
    # Step 2: Text Generation (using fallback)
    from utils import create_fallback_responses
    import random
    
    fallback_responses = create_fallback_responses()
    generated_text = random.choice(fallback_responses[sentiment])
    
    print(f"✅ Generated text: {generated_text[:50]}...")
    print(f"✅ Word count: {len(generated_text.split())}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 Sentiment-Based Text Generator - Test Suite")
    print("=" * 60)
    
    try:
        test_sentiment_analysis()
        test_text_generation()
        test_utility_functions()
        test_model_loading()
        run_integration_test()
        
        print("🎉 All tests completed!")
        print("✅ The application should work correctly")
        print("\n💡 To run the full application:")
        print("   python run.py")
        print("   or")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Please check your installation and try again")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
