"""
Utility functions for the Sentiment-Based Text Generator
"""

import re
import random
from typing import Dict, List, Tuple
import numpy as np

def clean_text(text: str) -> str:
    """
    Clean and normalize text input
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    
    # Ensure proper sentence structure
    text = text.strip()
    
    return text

def calculate_sentiment_score(sentiment_data: Dict) -> float:
    """
    Calculate a normalized sentiment score
    
    Args:
        sentiment_data: Dictionary containing sentiment analysis results
        
    Returns:
        Normalized score between -1 and 1
    """
    label = sentiment_data.get('label', 'NEUTRAL')
    score = sentiment_data.get('score', 0.5)
    
    if label == 'POSITIVE':
        return score
    elif label == 'NEGATIVE':
        return -score
    else:
        return 0.0

def generate_sentiment_keywords(sentiment: str) -> List[str]:
    """
    Generate keywords based on sentiment for better text generation
    
    Args:
        sentiment: Sentiment label (POSITIVE, NEGATIVE, NEUTRAL)
        
    Returns:
        List of sentiment-appropriate keywords
    """
    keywords = {
        'POSITIVE': [
            'amazing', 'wonderful', 'fantastic', 'excellent', 'great', 'awesome',
            'beautiful', 'incredible', 'outstanding', 'brilliant', 'perfect',
            'joyful', 'happy', 'delighted', 'pleased', 'satisfied', 'grateful'
        ],
        'NEGATIVE': [
            'terrible', 'awful', 'horrible', 'disappointing', 'frustrating',
            'upsetting', 'concerning', 'worrisome', 'difficult', 'challenging',
            'sad', 'angry', 'annoyed', 'disappointed', 'frustrated', 'upset'
        ],
        'NEUTRAL': [
            'interesting', 'notable', 'significant', 'important', 'relevant',
            'noteworthy', 'considerable', 'substantial', 'meaningful', 'valuable',
            'informative', 'educational', 'instructive', 'helpful', 'useful'
        ]
    }
    
    return keywords.get(sentiment, keywords['NEUTRAL'])

def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate user input text
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text"
    
    if len(text.strip()) < 3:
        return False, "Text must be at least 3 characters long"
    
    if len(text) > 1000:
        return False, "Text must be less than 1000 characters"
    
    return True, ""

def format_generated_text(text: str, max_length: int = 200) -> str:
    """
    Format generated text for better presentation
    
    Args:
        text: Generated text
        max_length: Maximum length for truncation
        
    Returns:
        Formatted text
    """
    # Clean the text
    text = clean_text(text)
    
    # Truncate if too long
    if len(text) > max_length:
        # Find the last complete sentence
        sentences = text.split('.')
        truncated = ''
        for sentence in sentences:
            if len(truncated + sentence + '.') <= max_length:
                truncated += sentence + '.'
            else:
                break
        text = truncated or text[:max_length] + '...'
    
    # Ensure proper capitalization
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    return text

def create_sentiment_visualization(sentiment_data: Dict) -> Dict:
    """
    Create data for sentiment visualization
    
    Args:
        sentiment_data: Sentiment analysis results
        
    Returns:
        Dictionary with visualization data
    """
    label = sentiment_data.get('label', 'NEUTRAL')
    confidence = sentiment_data.get('confidence', 0.5)
    
    # Color mapping
    colors = {
        'POSITIVE': '#28a745',
        'NEGATIVE': '#dc3545',
        'NEUTRAL': '#ffc107'
    }
    
    # Emoji mapping
    emojis = {
        'POSITIVE': '😊',
        'NEGATIVE': '😞',
        'NEUTRAL': '😐'
    }
    
    return {
        'label': label,
        'confidence': confidence,
        'color': colors.get(label, '#6c757d'),
        'emoji': emojis.get(label, '🤔'),
        'score': calculate_sentiment_score(sentiment_data)
    }

def get_length_settings(length_preference: str) -> Dict:
    """
    Get text generation settings based on length preference
    
    Args:
        length_preference: User's length preference
        
    Returns:
        Dictionary with generation settings
    """
    settings = {
        "Short (50-100 words)": {
            'max_length': 75,
            'min_length': 50,
            'temperature': 0.7
        },
        "Medium (100-200 words)": {
            'max_length': 150,
            'min_length': 100,
            'temperature': 0.8
        },
        "Long (200+ words)": {
            'max_length': 250,
            'min_length': 200,
            'temperature': 0.9
        }
    }
    
    return settings.get(length_preference, settings["Medium (100-200 words)"])

def create_fallback_responses() -> Dict[str, List[str]]:
    """
    Create fallback responses for when the main model fails
    
    Returns:
        Dictionary of fallback responses by sentiment
    """
    return {
        'POSITIVE': [
            "This is such a wonderful and uplifting experience! The positive energy and joy from this situation is truly inspiring and brings happiness to everyone involved.",
            "What an amazing and fantastic opportunity! This kind of positive development is exactly what makes life beautiful and worth celebrating.",
            "This is absolutely brilliant and outstanding! The incredible progress and positive outcomes here are truly remarkable and deserve recognition."
        ],
        'NEGATIVE': [
            "This is indeed a challenging and difficult situation that requires careful consideration and thoughtful approach to find the best possible solution.",
            "While this presents some real concerns and obstacles, it's important to remember that every challenge is also an opportunity for growth and learning.",
            "This is a tough situation that requires patience and perseverance. Remember that difficult times don't last forever, and there are always ways forward."
        ],
        'NEUTRAL': [
            "This is an interesting and noteworthy situation that presents several important considerations and factors to evaluate carefully.",
            "This matter requires balanced analysis and thoughtful consideration of all relevant aspects and potential implications.",
            "This is a significant development that warrants careful examination and objective evaluation of the various factors involved."
        ]
    }
