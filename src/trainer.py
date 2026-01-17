from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_veritas_model_v2(train_df):
    # 1. Advanced Vectorization: Unigrams + Bigrams
    # max_df=0.7 means ignore words that appear in more than 70% of news (too common)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7)
    
    # 2. Base Models
    lr = LogisticRegression(solver='liblinear', C=1.0)
    pac = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    
    # 3. Ensemble: Soft Voting takes the average of probabilities
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('pac', pac)],
        voting='soft' 
    )
    
    # Note: PAC doesn't support predict_proba by default. 
    # For 'soft' voting in this specific case, we'll stick to PAC's logic 
    # but wrap it or use 'hard' voting. Let's use 'hard' voting for maximum efficiency.
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('pac', pac)],
        voting='hard'
    )
    
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', ensemble)
    ])
    
    pipeline.fit(train_df['statement'], train_df['binary_label'])
    joblib.dump(pipeline, 'models/veritas_model.sav')
    return pipeline
