

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ComparativeAnalysis:
    def __init__(self, baseline_folder='baseline_results'):
        self.baseline_folder = baseline_folder
        self.models = {}
        
    def load_model_results(self, model_key):
        """Load a model's results"""
        file_path = f"{self.baseline_folder}/{model_key}_results.csv"
        df = pd.read_csv(file_path)
        self.models[model_key] = df
        print(f"✓ Loaded {model_key}: {len(df)} tweets")
        return df
    
    def analyze_distributions(self):
        """Analyze sentiment distributions across models"""
        print(f"\n{'='*70}")
        print("SENTIMENT DISTRIBUTION ANALYSIS")
        print(f"{'='*70}")
        
        dist_data = []
        
        for model_key, df in self.models.items():
            sentiment_col = f'{model_key}_sentiment'
            if sentiment_col in df.columns:
                dist = df[sentiment_col].value_counts()
                total = len(df)
                
                dist_data.append({
                    'Model': model_key,
                    'Positive_Count': dist.get('positive', 0),
                    'Neutral_Count': dist.get('neutral', 0),
                    'Negative_Count': dist.get('negative', 0),
                    'Positive_Pct': (dist.get('positive', 0) / total) * 100,
                    'Neutral_Pct': (dist.get('neutral', 0) / total) * 100,
                    'Negative_Pct': (dist.get('negative', 0) / total) * 100
                })
        
        dist_df = pd.DataFrame(dist_data)
        
        print("\n")
        print(dist_df[['Model', 'Positive_Pct', 'Neutral_Pct', 'Negative_Pct']].to_string(index=False))
        
        return dist_df
    
    def plot_distributions(self, output_file='distribution_comparison.png'):
        """Plot sentiment distributions"""
        dist_data = []
        
        for model_key, df in self.models.items():
            sentiment_col = f'{model_key}_sentiment'
            if sentiment_col in df.columns:
                dist = df[sentiment_col].value_counts(normalize=True) * 100
                dist_data.append({
                    'Model': model_key,
                    'Positive': dist.get('positive', 0),
                    'Neutral': dist.get('neutral', 0),
                    'Negative': dist.get('negative', 0)
                })
        
        dist_df = pd.DataFrame(dist_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(dist_df))
        width = 0.25
        
        ax.bar(x - width, dist_df['Positive'], width, label='Positive', color='#2ecc71')
        ax.bar(x, dist_df['Neutral'], width, label='Neutral', color='#95a5a6')
        ax.bar(x + width, dist_df['Negative'], width, label='Negative', color='#e74c3c')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment Distribution Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels(dist_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved distribution plot: {output_file}")
        plt.show()
    
    def analyze_by_athlete(self):
        """Analyze sentiment by athlete"""
        print(f"\n{'='*70}")
        print("SENTIMENT BY ATHLETE")
        print(f"{'='*70}")
        
        for model_key, df in self.models.items():
            sentiment_col = f'{model_key}_sentiment'
            if sentiment_col not in df.columns:
                continue
            
            print(f"\n{model_key}:")
            
            athlete_sentiment = df.groupby('athlete_display_name')[sentiment_col].value_counts(normalize=True) * 100
            print(athlete_sentiment.round(1))
    
    def agreement_analysis(self, model1_key, model2_key):
        """Analyze agreement between two models"""
        print(f"\n{'='*70}")
        print(f"AGREEMENT: {model1_key} vs {model2_key}")
        print(f"{'='*70}")
        
        df1 = self.models[model1_key]
        df2 = self.models[model2_key]
        
        # Merge on tweet_id
        merged = df1.merge(
            df2[['tweet_id', f'{model2_key}_sentiment']],
            on='tweet_id',
            how='inner'
        )
        
        sent1 = merged[f'{model1_key}_sentiment']
        sent2 = merged[f'{model2_key}_sentiment']
        
        # Calculate agreement
        agreement = (sent1 == sent2).sum()
        total = len(merged)
        agreement_pct = (agreement / total) * 100
        
        print(f"\nOverall Agreement: {agreement}/{total} ({agreement_pct:.1f}%)")
        
        # Per-label agreement
        print(f"\nAgreement by Label:")
        for label in ['positive', 'neutral', 'negative']:
            both_label = ((sent1 == label) & (sent2 == label)).sum()
            either_label = ((sent1 == label) | (sent2 == label)).sum()
            if either_label > 0:
                pct = (both_label / either_label) * 100
                print(f"  {label:8s}: {both_label}/{either_label} ({pct:.1f}%)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STEP 3B: COMPARATIVE ANALYSIS (NO HUMAN ANNOTATIONS)")
    print("="*70)
    
    # Initialize
    analyzer = ComparativeAnalysis()
    
    # Load model results
    analyzer.load_model_results('xlm-roberta')
    analyzer.load_model_results('mbert')
    
    # Analyze distributions
    analyzer.analyze_distributions()
    analyzer.plot_distributions()
    
    # Analyze by athlete
    analyzer.analyze_by_athlete()
    
    # Agreement between models
    analyzer.agreement_analysis('xlm-roberta', 'mbert')
    
    print("\n" + "="*70)
    print("✓ COMPARATIVE ANALYSIS COMPLETE!")
    print("="*70)