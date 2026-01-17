from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_bias_and_propaganda(text):
    # 1. Subjectivity Analysis (0.0 = Objective, 1.0 = Highly Subjective/Opinionated)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    
    # 2. Emotional Intensity (VADER)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    intensity = vader_scores['compound']
    
    # 3. Propaganda Logic
    # High Subjectivity + High Emotional Intensity usually indicates Propaganda
    propaganda_risk = "LOW"
    if subjectivity > 0.6 and abs(intensity) > 0.5:
        propaganda_risk = "HIGH"
    elif subjectivity > 0.4 or abs(intensity) > 0.3:
        propaganda_risk = "MODERATE"
        
    # 4. Political Bias Mapping
    # Negative intensity often correlates with attack-style political bias
    bias_direction = "Neutral/Balanced"
    if intensity > 0.4:
        bias_direction = "Strongly Positive/Promotional"
    elif intensity < -0.4:
        bias_direction = "Strongly Negative/Critical"

    return {
        "subjectivity_score": f"{subjectivity:.2f}",
        "emotional_intensity": f"{intensity:.2f}",
        "propaganda_risk": propaganda_risk,
        "bias_lean": bias_direction
    }
