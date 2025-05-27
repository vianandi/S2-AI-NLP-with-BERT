import pandas as pd
import re
import csv

def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='utf-8', sep='\t', on_bad_lines='skip')  

    df = df[['Tweet', 'sentiment']].dropna()
    
    def clean_text(text):
        text = str(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.lower().strip()

    df['text'] = df['Tweet'].apply(clean_text)
    
    if df['sentiment'].dtype == 'object':
        df['label'] = df['sentiment'].map({'positif': 1, 'negatif': 0})
    else:
        df['label'] = df['sentiment'].map({1: 1, 0: 0})
        
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df.sample(n=5000, random_state=42) 
    
    print(f"Original sentiment values: {df['sentiment'].unique()}")
    print(f"Mapped labels: {df['label'].unique()}")
    
    return df