import pandas as pd

def load_liar_data(file_path):
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 
        'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context'
    ]
    df = pd.read_csv(file_path, sep='\t', names=columns, on_bad_lines='skip')
    
    # ENRICHMENT: Replace 'statement' with the combined metadata string
    # We keep the name 'statement' so the rest of the code doesn't break
    df['statement'] = df['speaker'].fillna('Unknown') + " (" + df['party'].fillna('None') + "): " + df['statement']
    
    mapping = {
        'true': 1, 'mostly-true': 1, 'half-true': 1,
        'barely-true': 0, 'false': 0, 'pants-fire': 0
    }
    df['binary_label'] = df['label'].map(mapping)
    return df[['statement', 'binary_label']].dropna()
