from newspaper import Article
import joblib
from src.propaganda_engine import analyze_bias_and_propaganda

def analyze_url_v3(url):
    try:
        # 1. Scrape & Summarize
        article = Article(url)
        article.download(); article.parse(); article.nlp()
        
        # 2. Get Propaganda & Bias Insights
        bias_data = analyze_bias_and_propaganda(article.text)
        
        # 3. Veritas ML Verdict
        model = joblib.load('models/veritas_model.sav')
        source_name = url.split('//')[-1].split('/')[0]
        enriched_content = f"{source_name} (News): {article.title}. {article.text}"
        prediction = model.predict([enriched_content])[0]
        verdict = "REAL" if prediction == 1 else "FAKE"
        
        return {
            "title": article.title,
            "summary": article.summary,
            "verdict": verdict,
            "bias_metrics": bias_data
        }
    except Exception as e:
        return {"error": str(e)}
