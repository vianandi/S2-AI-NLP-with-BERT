import pandas as pd
import os

def test_data_loading():
    """Test loading sentiment data files"""
    data_dir = "sentiment_ablation/data"
    
    files = [
        "dataset_tweet_sentimen_tayangan_tv.csv",
        "dataset_tweet_sentiment_opini_film.csv", 
        "Dataset Sentimen kurikulum 2013.xlsx",
        "id-tourism-sentimentanalysis.xlsx"
    ]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
            try:
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                    
                print(f"   Shape: {data.shape}")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Sample data:")
                print(data.head(2))
                print("-" * 50)
                
            except Exception as e:
                print(f"   ❌ Error loading: {str(e)}")
        else:
            print(f"❌ Not found: {file}")

if __name__ == "__main__":
    test_data_loading()