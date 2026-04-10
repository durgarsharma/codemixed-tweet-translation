
!pip install transformers torch pandas tqdm scikit-learn -q


print("UPLOAD DATA FILE")

from google.colab import files
import pandas as pd

uploaded = files.upload()

data_file = list(uploaded.keys())[0]
print(f"\n✓ Uploaded: {data_file}")

print("\nLoading data with robust parsing...")

try:
  
    df = pd.read_csv(
        data_file,
        encoding='utf-8',
        on_bad_lines='skip',  
        engine='python',     
        quoting=1,            
        escapechar='\\'       
    )
    print(f" Loaded {len(df)} tweets (method 1)")
except Exception as e:
    print(f"Method 1 failed: {e}")
    print("Trying alternative parsing...")

    try:
        
        df = pd.read_csv(
            data_file,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            sep=',',
            quotechar='"',
            doublequote=True,
            error_bad_lines=False
        )
        print(f"✓ Loaded {len(df)} tweets (method 2)")
    except Exception as e:
        print(f"Method 2 failed: {e}")
        print("Trying chunk-based loading...")

        chunks = []
        chunk_size = 1000

        for chunk in pd.read_csv(
            data_file,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            chunksize=chunk_size
        ):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        print(f"✓ Loaded {len(df)} tweets (method 3 - chunks)")

print(f"\n Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(3))


df = df.dropna(subset=['tweet_id', 'raw_content'])

if 'raw_content' in df.columns:
    df['raw_content'] = df['raw_content'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ')

if 'translated_content' in df.columns:
    df['translated_content'] = df['translated_content'].astype(str).str.replace('\r', ' ').str.replace('\n', ' ')

print(f" Final dataset: {len(df)} tweets")
print(f" Columns: {df.columns.tolist()}")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from tqdm import tqdm
import warnings
import json
from datetime import datetime
import numpy as np
warnings.filterwarnings('ignore')

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU ' if device == 0 else 'CPU (slower)'}")
if device == 0:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

class MultilingualSentimentBaselines:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        print(f"Loaded {len(self.df)} tweets")

        self.models = {}
        self.results = {}

    def load_model(self, model_name, model_key):
        print(f"Loading {model_key}...")
        print(f"Model: {model_name}")

        try:
            if model_key == 'xlm-roberta':
                self.models[model_key] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=device,
                    max_length=512,
                    truncation=True
                )
                print(f" {model_key} loaded successfully")

            elif model_key == 'mbert':
                self.models[model_key] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=device,
                    max_length=512,
                    truncation=True
                )
                print(f"{model_key} loaded successfully")

            elif model_key == 'bert-sentiment':
                self.models[model_key] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=device,
                    max_length=512,
                    truncation=True
                )
                print(f" {model_key} loaded successfully")

        except Exception as e:
            print(f"✗ Error loading {model_key}: {str(e)}")
            return False

        return True

    def normalize_sentiment(self, label, score, model_key):
        label = label.lower()

        if model_key == 'xlm-roberta':
            if 'positive' in label:
                return 'positive'
            elif 'negative' in label:
                return 'negative'
            else:
                return 'neutral'

        elif model_key == 'mbert':
            if '1' in label or '2' in label:
                return 'negative'
            elif '3' in label:
                return 'neutral'
            elif '4' in label or '5' in label:
                return 'positive'

        elif model_key == 'bert-sentiment':
            if 'label_0' in label or 'negative' in label:
                return 'negative'
            elif 'label_2' in label or 'positive' in label:
                return 'positive'
            else:
                return 'neutral'

        return 'neutral'

    def run_baseline(self, model_key, text_column='raw_content', sample_size=None):

        if model_key not in self.models:
            print(f"Model {model_key} not loaded, skipping...")
            return None

        print(f"Running {model_key} on {text_column}...")
   
        if sample_size:
            df_subset = self.df.sample(min(sample_size, len(self.df)), random_state=42)
            print(f"Using sample of {len(df_subset)} tweets")
        else:
            df_subset = self.df.copy()

        results = []

        batch_size = 16  
        texts = df_subset[text_column].fillna('').tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc=f"{model_key}"):
            batch = texts[i:i+batch_size]
            try:
                predictions = self.models[model_key](batch)
                for pred in predictions:
                    normalized = self.normalize_sentiment(pred['label'], pred['score'], model_key)
                    results.append({
                        'label': normalized,
                        'score': pred['score'],
                        'original_label': pred['label']
                    })
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                results.extend([{'label': 'neutral', 'score': 0.0, 'original_label': 'error'}] * len(batch))

        df_subset[f'{model_key}_sentiment'] = [r['label'] for r in results]
        df_subset[f'{model_key}_score'] = [r['score'] for r in results]
        df_subset[f'{model_key}_original'] = [r['original_label'] for r in results]

        self.results[model_key] = df_subset

        print(f"\n {model_key} complete!")
        self._print_distribution(df_subset[f'{model_key}_sentiment'])

        return df_subset

    def _print_distribution(self, sentiment_series):
        dist = sentiment_series.value_counts()
        total = len(sentiment_series)

        print("\nSentiment Distribution:")
        for label in ['positive', 'neutral', 'negative']:
            if label in dist:
                count = dist[label]
                pct = (count / total) * 100
                print(f"  {label:8s}: {count:5d} ({pct:5.2f}%)")

    def compare_all_models(self):
        print("COMPARISON OF ALL MODELS")

        comparison_data = []

        for model_key in self.results:
            df = self.results[model_key]
            sentiment_col = f'{model_key}_sentiment'

            if sentiment_col in df.columns:
                dist = df[sentiment_col].value_counts()
                total = len(df)

                comparison_data.append({
                    'Model': model_key,
                    'Positive': f"{dist.get('positive', 0)} ({dist.get('positive', 0)/total*100:.1f}%)",
                    'Neutral': f"{dist.get('neutral', 0)} ({dist.get('neutral', 0)/total*100:.1f}%)",
                    'Negative': f"{dist.get('negative', 0)} ({dist.get('negative', 0)/total*100:.1f}%)",
                    'Total': total
                })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def save_results(self):
        import os
        os.makedirs('baseline_results', exist_ok=True)

        for model_key, df in self.results.items():
            output_file = f"baseline_results/{model_key}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Saved {output_file}")

        comparison = self.compare_all_models()
        comparison.to_csv("baseline_results/model_comparison.csv", index=False)
        print(f" Saved baseline_results/model_comparison.csv")

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tweets': len(self.df),
            'models_evaluated': list(self.results.keys()),
        }

        with open("baseline_results/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved baseline_results/summary.json")

        return comparison

print("INITIALIZING BASELINE MODELS")

baselines = MultilingualSentimentBaselines(df)

print("\n Loading XLM-RoBERTa (best for code-mixed)...")
baselines.load_model("cardiffnlp/twitter-xlm-roberta-base-sentiment", "xlm-roberta")

print("\n Loading mBERT (multilingual)")
baselines.load_model("nlptown/bert-base-multilingual-uncased-sentiment", "mbert")

print("\n Loading English BERT baseline")
baselines.load_model("cardiffnlp/twitter-roberta-base-sentiment-latest", "bert-sentiment")

print("\n All models loaded!")

print("RUNNING XLM-ROBERTA ON RAW (CODE-MIXED) CONTENT")


baselines.run_baseline('xlm-roberta', text_column='raw_content')

print("RUNNING mBERT ON RAW (CODE-MIXED) CONTENT")

baselines.run_baseline('mbert', text_column='raw_content')

if 'translated_content' in df.columns:
    print("RUNNING XLM-ROBERTA ON TRANSLATED CONTENT")

    baselines.models['xlm-roberta-translated'] = baselines.models['xlm-roberta']
    baselines.run_baseline('xlm-roberta-translated', text_column='translated_content')

    print("RUNNING BERT-SENTIMENT ON TRANSLATED CONTENT")

    baselines.models['bert-sentiment-translated'] = baselines.models['bert-sentiment']
    baselines.run_baseline('bert-sentiment-translated', text_column='translated_content')
else:
    print("\n No translated_content column found, skipping translated analysis")

print("FINAL COMPARISON")

comparison_df = baselines.compare_all_models()

print("SAVING RESULTS")

baselines.save_results()

!zip -r baseline_results.zip baseline_results/

print("\nDownloading results...")

from google.colab import files
files.download('baseline_results.zip')


print("SAMPLE RESULTS")

sample_results = baselines.results['xlm-roberta'].sample(5)
cols_to_show = ['tweet_id', 'athlete_display_name', 'raw_content']

for model_key in baselines.results.keys():
    if f'{model_key}_sentiment' in baselines.results[model_key].columns:
        sample_results = sample_results.merge(
            baselines.results[model_key][['tweet_id', f'{model_key}_sentiment']],
            on='tweet_id',
            how='left'
        )

print("\nSample predictions from all models:")
print(sample_results[['raw_content'] + [col for col in sample_results.columns if '_sentiment' in col]].to_string())

print("DIAGNOSING TRANSLATED MODELS ISSUE")

print("\nSample translated_content:")
print(df['translated_content'].head(10).tolist())

print("\nTranslated content stats:")
print(f"Total: {len(df)}")
print(f"Non-null: {df['translated_content'].notna().sum()}")
print(f"Empty strings: {(df['translated_content'] == '').sum()}")
print(f"NaN values: {df['translated_content'].isna().sum()}")

print("\nChecking actual model outputs...")
xlm_trans_results = pd.read_csv('baseline_results/xlm-roberta-translated_results.csv')
print("\nOriginal labels from XLM-RoBERTa-translated:")
print(xlm_trans_results['xlm-roberta-translated_original'].value_counts())
