from sklearn.metrics import classification_report, accuracy_score
import joblib

def evaluate_veritas_model(test_df):
    model = joblib.load('models/veritas_model.sav')
    
    # The data_loader now ensures 'statement' is the enriched column
    y_true = test_df['binary_label']
    y_pred = model.predict(test_df['statement'])
    
    print("--- Veritas AI V2 Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}\n")
    print(classification_report(y_true, y_pred))
