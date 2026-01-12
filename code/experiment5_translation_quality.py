

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

class TranslationQualityAnalysis:
    def __init__(self):
        print("Loading baseline results...")
        
        # Load results with both raw and translated content
        self.xlm_raw = pd.read_csv('baseline_results/xlm-roberta_results.csv')
        
        # Try to load translated results
        try:
            self.xlm_trans = pd.read_csv('baseline_results/xlm-roberta-translated_results_fixed.csv')
            self.has_translations = True
            print("✓ Found translated results")
        except:
            print("⚠ No fixed translated results found, will analyze existing data")
            self.has_translations = False
        
        print(f"✓ Loaded {len(self.xlm_raw)} tweets")
    
    def detect_language(self, text):
        """Detect language type"""
        if pd.isna(text):
            return 'unknown'
        text = str(text)
        has_hindi = bool(re.search(r'[\u0900-\u097F]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        if has_hindi and has_english:
            return 'code-mixed'
        elif has_hindi:
            return 'hindi'
        elif has_english:
            return 'english'
        return 'other'
    
    def analyze_translation_impact(self):
        """Analyze how translation affects sentiment predictions"""
        print("\n" + "="*70)
        print("TRANSLATION IMPACT ANALYSIS")
        print("="*70)
        
        if not self.has_translations:
            print("\n⚠ Skipping - no translated results available")
            return
        
        # Merge raw and translated results
        merged = self.xlm_raw.merge(
            self.xlm_trans[['tweet_id', 'xlm-roberta-translated_sentiment']],
            on='tweet_id',
            how='inner'
        )
        
        # Add language detection
        merged['language_type'] = merged['raw_content'].apply(self.detect_language)
        
        # Focus on code-mixed tweets
        code_mixed = merged[merged['language_type'] == 'code-mixed']
        
        print(f"\nAnalyzing {len(code_mixed)} code-mixed tweets")
        
        # Calculate agreement
        agreement = (code_mixed['xlm-roberta_sentiment'] == code_mixed['xlm-roberta-translated_sentiment']).sum()
        total = len(code_mixed)
        
        print(f"\nAgreement between raw and translated:")
        print(f"  Same prediction: {agreement} / {total} ({agreement/total*100:.1f}%)")
        print(f"  Different prediction: {total-agreement} / {total} ({(total-agreement)/total*100:.1f}%)")
        
        # Sentiment shifts
        print(f"\nSentiment Shifts (Code-Mixed Tweets):")
        for orig_sent in ['positive', 'neutral', 'negative']:
            for trans_sent in ['positive', 'neutral', 'negative']:
                if orig_sent != trans_sent:
                    count = ((code_mixed['xlm-roberta_sentiment'] == orig_sent) & 
                            (code_mixed['xlm-roberta-translated_sentiment'] == trans_sent)).sum()
                    if count > 0:
                        print(f"  {orig_sent} → {trans_sent}: {count} ({count/total*100:.1f}%)")
    
    def analyze_length_statistics(self):
        """Analyze tweet length before and after translation"""
        print("\n" + "="*70)
        print("TWEET LENGTH ANALYSIS")
        print("="*70)
        
        # Calculate lengths
        self.xlm_raw['raw_length'] = self.xlm_raw['raw_content'].str.len()
        
        if 'translated_content' in self.xlm_raw.columns:
            self.xlm_raw['trans_length'] = self.xlm_raw['translated_content'].fillna('').str.len()
            
            # Filter for code-mixed
            self.xlm_raw['language_type'] = self.xlm_raw['raw_content'].apply(self.detect_language)
            code_mixed = self.xlm_raw[self.xlm_raw['language_type'] == 'code-mixed']
            
            print(f"\nLength Statistics for Code-Mixed Tweets (n={len(code_mixed)}):")
            print(f"  Original:")
            print(f"    Mean: {code_mixed['raw_length'].mean():.1f} characters")
            print(f"    Median: {code_mixed['raw_length'].median():.1f} characters")
            
            print(f"  Translated:")
            print(f"    Mean: {code_mixed['trans_length'].mean():.1f} characters")
            print(f"    Median: {code_mixed['trans_length'].median():.1f} characters")
            
            # Length change
            code_mixed['length_change'] = code_mixed['trans_length'] - code_mixed['raw_length']
            print(f"\n  Average length change: {code_mixed['length_change'].mean():.1f} characters")
    
    def analyze_sentiment_consistency(self):
        """Analyze which sentiments are most affected by language"""
        print("\n" + "="*70)
        print("SENTIMENT CONSISTENCY BY LANGUAGE TYPE")
        print("="*70)
        
        self.xlm_raw['language_type'] = self.xlm_raw['raw_content'].apply(self.detect_language)
        
        print("\nSentiment Distribution:")
        for lang in ['english', 'code-mixed', 'hindi']:
            lang_tweets = self.xlm_raw[self.xlm_raw['language_type'] == lang]
            if len(lang_tweets) > 0:
                dist = lang_tweets['xlm-roberta_sentiment'].value_counts(normalize=True) * 100
                print(f"\n{lang.upper()} (n={len(lang_tweets)}):")
                for sent in ['positive', 'neutral', 'negative']:
                    if sent in dist:
                        print(f"  {sent:8s}: {dist[sent]:.1f}%")
    
    def sample_translations(self, n=10):
        """Show sample translations for code-mixed tweets"""
        print("\n" + "="*70)
        print(f"SAMPLE TRANSLATIONS (showing {n} examples)")
        print("="*70)
        
        if 'translated_content' not in self.xlm_raw.columns:
            print("\n⚠ No translated_content column found")
            return
        
        self.xlm_raw['language_type'] = self.xlm_raw['raw_content'].apply(self.detect_language)
        code_mixed = self.xlm_raw[self.xlm_raw['language_type'] == 'code-mixed']
        
        if len(code_mixed) > 0:
            samples = code_mixed.sample(min(n, len(code_mixed)), random_state=42)
            
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                print(f"\n{i}. ORIGINAL: {row['raw_content'][:120]}...")
                print(f"   TRANSLATED: {str(row['translated_content'])[:120]}...")
                print(f"   SENTIMENT: {row['xlm-roberta_sentiment']}")
    
    def plot_translation_analysis(self):
        """Create visualization"""
        if not self.has_translations:
            print("\n⚠ Skipping plots - no translation data")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Agreement matrix
        merged = self.xlm_raw.merge(
            self.xlm_trans[['tweet_id', 'xlm-roberta-translated_sentiment']],
            on='tweet_id',
            how='inner'
        )
        merged['language_type'] = merged['raw_content'].apply(self.detect_language)
        code_mixed = merged[merged['language_type'] == 'code-mixed']
        
        if len(code_mixed) > 0:
            # Confusion matrix
            labels = ['positive', 'neutral', 'negative']
            confusion = np.zeros((3, 3))
            label_to_idx = {'positive': 0, 'neutral': 1, 'negative': 2}
            
            for _, row in code_mixed.iterrows():
                i = label_to_idx[row['xlm-roberta_sentiment']]
                j = label_to_idx[row['xlm-roberta-translated_sentiment']]
                confusion[i, j] += 1
            
            confusion_pct = confusion / confusion.sum() * 100
            
            sns.heatmap(
                confusion_pct,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=['Pos', 'Neu', 'Neg'],
                yticklabels=['Pos', 'Neu', 'Neg'],
                ax=axes[0],
                cbar_kws={'label': 'Percentage (%)'}
            )
            axes[0].set_xlabel('Translated Sentiment', fontsize=12)
            axes[0].set_ylabel('Raw Sentiment', fontsize=12)
            axes[0].set_title('Translation Impact on Code-Mixed Tweets', fontsize=12, fontweight='bold')
        
        # Sentiment by language type
        self.xlm_raw['language_type'] = self.xlm_raw['raw_content'].apply(self.detect_language)
        
        lang_sentiment = []
        for lang in ['english', 'code-mixed']:
            lang_data = self.xlm_raw[self.xlm_raw['language_type'] == lang]
            if len(lang_data) > 0:
                dist = lang_data['xlm-roberta_sentiment'].value_counts(normalize=True) * 100
                lang_sentiment.append({
                    'Language': lang.replace('-', ' ').title(),
                    'Positive': dist.get('positive', 0),
                    'Neutral': dist.get('neutral', 0),
                    'Negative': dist.get('negative', 0)
                })
        
        if lang_sentiment:
            df_plot = pd.DataFrame(lang_sentiment)
            x = np.arange(len(df_plot))
            width = 0.25
            
            axes[1].bar(x - width, df_plot['Positive'], width, label='Positive', color='#2ecc71')
            axes[1].bar(x, df_plot['Neutral'], width, label='Neutral', color='#95a5a6')
            axes[1].bar(x + width, df_plot['Negative'], width, label='Negative', color='#e74c3c')
            
            axes[1].set_ylabel('Percentage (%)', fontsize=12)
            axes[1].set_title('Sentiment by Language Type', fontsize=12, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(df_plot['Language'])
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('paper_figures/fig_translation_quality.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: paper_figures/fig_translation_quality.png")
        plt.close()

if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 5: TRANSLATION QUALITY ASSESSMENT")
    print("="*70)
    
    analyzer = TranslationQualityAnalysis()
    analyzer.analyze_translation_impact()
    analyzer.analyze_length_statistics()
    analyzer.analyze_sentiment_consistency()
    analyzer.sample_translations(n=5)
    analyzer.plot_translation_analysis()
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT 5 COMPLETE!")
    print("="*70)