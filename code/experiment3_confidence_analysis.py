

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv

class ConfidenceAnalysis:
    def __init__(self):
        print("Loading data...")
        
        # Load baseline results
        self.xlm_df = pd.read_csv('baseline_results/xlm-roberta_results.csv')
        self.mbert_df = pd.read_csv('baseline_results/mbert_results.csv')
        
        print(f"✓ Loaded {len(self.xlm_df)} XLM-RoBERTa predictions")
        print(f"✓ Loaded {len(self.mbert_df)} mBERT predictions")
        
        # Detect language type directly from raw_content in baseline results
        print("Detecting language types...")
        self.xlm_df['language_type'] = self.xlm_df['raw_content'].apply(self.detect_language)
        self.mbert_df['language_type'] = self.mbert_df['raw_content'].apply(self.detect_language)
        
        print("✓ Language detection complete")
    
    def detect_language(self, text):
        """Detect if text contains Hindi (Devanagari script)"""
        if pd.isna(text):
            return 'unknown'
        
        text = str(text)
        
        # Check for Devanagari Unicode range
        has_hindi = bool(re.search(r'[\u0900-\u097F]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        if has_hindi and has_english:
            return 'code-mixed'
        elif has_hindi:
            return 'hindi'
        elif has_english:
            return 'english'
        else:
            return 'other'
    
    def analyze_confidence_by_language(self):
        """Compare confidence scores between English and code-mixed"""
        print("\n" + "="*70)
        print("CONFIDENCE ANALYSIS BY LANGUAGE TYPE")
        print("="*70)
        
        results = []
        
        for model_name, df, score_col in [
            ('XLM-RoBERTa', self.xlm_df, 'xlm-roberta_score'),
            ('mBERT', self.mbert_df, 'mbert_score')
        ]:
            print(f"\n{model_name}:")
            
            english = df[df['language_type'] == 'english'][score_col].dropna()
            code_mixed = df[df['language_type'] == 'code-mixed'][score_col].dropna()
            
            print(f"  English tweets (n={len(english)}):")
            print(f"    Mean confidence: {english.mean():.4f}")
            print(f"    Median confidence: {english.median():.4f}")
            print(f"    Std: {english.std():.4f}")
            
            print(f"  Code-mixed tweets (n={len(code_mixed)}):")
            print(f"    Mean confidence: {code_mixed.mean():.4f}")
            print(f"    Median confidence: {code_mixed.median():.4f}")
            print(f"    Std: {code_mixed.std():.4f}")
            
            # T-test
            from scipy.stats import ttest_ind
            if len(english) > 0 and len(code_mixed) > 0:
                t_stat, p_val = ttest_ind(english, code_mixed)
                print(f"  T-test: t={t_stat:.4f}, p={p_val:.4e}")
                
                sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
                print(f"  Result: {sig} difference in confidence")
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(english)-1)*english.std()**2 + (len(code_mixed)-1)*code_mixed.std()**2) / (len(english)+len(code_mixed)-2))
                cohens_d = (english.mean() - code_mixed.mean()) / pooled_std
                print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
                
                results.append({
                    'model': model_name,
                    'english_mean': english.mean(),
                    'code_mixed_mean': code_mixed.mean(),
                    't_stat': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d
                })
        
        return results
    
    def analyze_confidence_by_sentiment(self):
        """Analyze confidence scores by sentiment class"""
        print("\n" + "="*70)
        print("CONFIDENCE BY SENTIMENT CLASS")
        print("="*70)
        
        for model_name, df, score_col, sent_col in [
            ('XLM-RoBERTa', self.xlm_df, 'xlm-roberta_score', 'xlm-roberta_sentiment'),
            ('mBERT', self.mbert_df, 'mbert_score', 'mbert_sentiment')
        ]:
            print(f"\n{model_name}:")
            
            for sentiment in ['positive', 'neutral', 'negative']:
                scores = df[df[sent_col] == sentiment][score_col].dropna()
                if len(scores) > 0:
                    print(f"  {sentiment:8s}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    def plot_confidence_comparison(self):
        """Visualize confidence by language type"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (model_name, df, score_col) in enumerate([
            ('XLM-RoBERTa', self.xlm_df, 'xlm-roberta_score'),
            ('mBERT', self.mbert_df, 'mbert_score')
        ]):
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            
            for lang_type in ['english', 'code-mixed']:
                scores = df[df['language_type'] == lang_type][score_col].dropna()
                if len(scores) > 0:
                    data_to_plot.append(scores)
                    labels.append(lang_type.replace('-', ' ').title())
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                colors = ['#3498db', '#e74c3c']
                for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_ylabel('Confidence Score', fontsize=12)
                ax.set_title(f'{model_name}\nConfidence by Language Type', 
                            fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('paper_figures/fig_confidence_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: paper_figures/fig_confidence_comparison.png")
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot distribution of confidence scores"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, (model_name, df, score_col) in enumerate([
            ('XLM-RoBERTa', self.xlm_df, 'xlm-roberta_score'),
            ('mBERT', self.mbert_df, 'mbert_score')
        ]):
            # English
            ax1 = axes[idx, 0]
            english_scores = df[df['language_type'] == 'english'][score_col].dropna()
            if len(english_scores) > 0:
                ax1.hist(english_scores, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
                ax1.axvline(english_scores.mean(), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {english_scores.mean():.3f}')
                ax1.set_xlabel('Confidence Score', fontsize=11)
                ax1.set_ylabel('Frequency', fontsize=11)
                ax1.set_title(f'{model_name} - English Tweets', fontsize=11, fontweight='bold')
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
            
            # Code-mixed
            ax2 = axes[idx, 1]
            code_mixed_scores = df[df['language_type'] == 'code-mixed'][score_col].dropna()
            if len(code_mixed_scores) > 0:
                ax2.hist(code_mixed_scores, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
                ax2.axvline(code_mixed_scores.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {code_mixed_scores.mean():.3f}')
                ax2.set_xlabel('Confidence Score', fontsize=11)
                ax2.set_ylabel('Frequency', fontsize=11)
                ax2.set_title(f'{model_name} - Code-Mixed Tweets', fontsize=11, fontweight='bold')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('paper_figures/fig_confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: paper_figures/fig_confidence_distribution.png")
        plt.close()
    
    def create_summary_table(self, results):
        """Create LaTeX table"""
        latex = "\\begin{table}[H]\n"
        latex += "    \\centering\n"
        latex += "    \\caption{Model Confidence: English vs. Code-Mixed Tweets}\n"
        latex += "    \\begin{tabular}{lccc}\n"
        latex += "    \\hline\n"
        latex += "    \\textbf{Model} & \\textbf{English} & \\textbf{Code-Mixed} & \\textbf{p-value} \\\\\n"
        latex += "    \\hline\n"
        
        for r in results:
            latex += f"    {r['model']} & {r['english_mean']:.3f} & {r['code_mixed_mean']:.3f} & {r['p_value']:.4f} \\\\\n"
        
        latex += "    \\hline\n"
        latex += "    \\end{tabular}\n"
        latex += "    \\label{tab:confidence_comparison}\n"
        latex += "\\end{table}"
        
        with open('paper_tables/table_confidence_comparison.tex', 'w') as f:
            f.write(latex)
        
        print("\n✓ Saved: paper_tables/table_confidence_comparison.tex")

if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 3: CONFIDENCE SCORE ANALYSIS")
    print("="*70)
    
    analyzer = ConfidenceAnalysis()
    results = analyzer.analyze_confidence_by_language()
    analyzer.analyze_confidence_by_sentiment()
    analyzer.plot_confidence_comparison()
    analyzer.plot_confidence_distribution()
    analyzer.create_summary_table(results)
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT 3 COMPLETE!")
    print("="*70)