

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

class LanguageMixingAnalysis:
    def __init__(self, data_file='all_tweets_combined.csv'):
        # Robust CSV loading
        print(f"Loading {data_file}...")
        try:
            self.df = pd.read_csv(
                data_file,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='python',
                quoting=1,
                escapechar='\\'
            )
        except Exception as e:
            print(f"Method 1 failed, trying alternative...")
            # Read line by line if needed
            import csv
            data_rows = []
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('tweet_id') and row.get('raw_content'):
                        data_rows.append(row)
            self.df = pd.DataFrame(data_rows)
        
        print(f"✓ Loaded {len(self.df)} tweets")
        
    def detect_language_simple(self, text):
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
    
    def calculate_mixing_ratio(self, text):
        """Calculate the ratio of Hindi to English words"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        
        # Count Devanagari characters
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total = hindi_chars + english_chars
        if total == 0:
            return 0
        
        return hindi_chars / total
    
    def analyze_dataset(self):
        """Analyze language composition of dataset"""
        print("\n" + "="*70)
        print("LANGUAGE COMPOSITION ANALYSIS")
        print("="*70)
        
        # Detect language for each tweet
        print("Detecting languages...")
        self.df['language_type'] = self.df['raw_content'].apply(self.detect_language_simple)
        self.df['hindi_ratio'] = self.df['raw_content'].apply(self.calculate_mixing_ratio)
        
        # Statistics
        lang_dist = self.df['language_type'].value_counts()
        total = len(self.df)
        
        print(f"\nLanguage Distribution:")
        for lang, count in lang_dist.items():
            pct = (count / total) * 100
            print(f"  {lang:12s}: {count:5d} ({pct:5.2f}%)")
        
        # Code-mixing intensity for code-mixed tweets
        code_mixed = self.df[self.df['language_type'] == 'code-mixed']
        if len(code_mixed) > 0:
            print(f"\nCode-Mixed Tweets Analysis:")
            print(f"  Total code-mixed: {len(code_mixed)}")
            print(f"  Mean Hindi ratio: {code_mixed['hindi_ratio'].mean():.2f}")
            print(f"  Median Hindi ratio: {code_mixed['hindi_ratio'].median():.2f}")
            print(f"  Std Hindi ratio: {code_mixed['hindi_ratio'].std():.2f}")
        
        return self.df
    
    def analyze_by_athlete(self):
        """Analyze language mixing by athlete"""
        print("\n" + "="*70)
        print("LANGUAGE MIXING BY ATHLETE")
        print("="*70)
        
        # Skip if no athlete column
        if 'athlete_display_name' not in self.df.columns:
            print("⚠ No athlete_display_name column found")
            return
        
        athlete_lang = self.df.groupby('athlete_display_name')['language_type'].value_counts(normalize=True) * 100
        print("\n")
        print(athlete_lang.round(1))
        
    def plot_language_distribution(self):
        """Visualize language distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        lang_counts = self.df['language_type'].value_counts()
        colors = {'code-mixed': '#e74c3c', 'english': '#3498db', 'hindi': '#2ecc71', 'other': '#95a5a6'}
        
        ax1.pie(
            lang_counts.values,
            labels=[f"{x.title()}\n({c})" for x, c in zip(lang_counts.index, lang_counts.values)],
            autopct='%1.1f%%',
            colors=[colors.get(x, '#95a5a6') for x in lang_counts.index],
            startangle=90
        )
        ax1.set_title('Language Distribution in Dataset', fontsize=14, fontweight='bold')
        
        # Hindi ratio distribution for code-mixed tweets
        code_mixed = self.df[self.df['language_type'] == 'code-mixed']
        if len(code_mixed) > 0:
            ax2.hist(code_mixed['hindi_ratio'], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax2.axvline(code_mixed['hindi_ratio'].mean(), color='black', linestyle='--', 
                       linewidth=2, label=f'Mean: {code_mixed["hindi_ratio"].mean():.2f}')
            ax2.set_xlabel('Hindi Character Ratio', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Code-Mixing Intensity Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No code-mixed tweets found', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('paper_figures/fig_language_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: paper_figures/fig_language_distribution.png")
        plt.close()
    
    def sample_tweets_by_type(self):
        """Show sample tweets for each language type"""
        print("\n" + "="*70)
        print("SAMPLE TWEETS BY LANGUAGE TYPE")
        print("="*70)
        
        for lang_type in ['code-mixed', 'english', 'hindi']:
            tweets = self.df[self.df['language_type'] == lang_type]
            if len(tweets) > 0:
                print(f"\n{lang_type.upper()} (showing 3 samples):")
                print("-" * 70)
                for _, tweet in tweets.sample(min(3, len(tweets)), random_state=42).iterrows():
                    print(f"  {tweet['raw_content'][:100]}...")
    
    def create_summary_table(self):
        """Create LaTeX table for paper"""
        lang_dist = self.df['language_type'].value_counts()
        total = len(self.df)
        
        latex = "\\begin{table}[H]\n"
        latex += "    \\centering\n"
        latex += "    \\caption{Language Composition of Tweet Dataset}\n"
        latex += "    \\begin{tabular}{lcc}\n"
        latex += "    \\hline\n"
        latex += "    \\textbf{Language Type} & \\textbf{Count} & \\textbf{Percentage (\\%)} \\\\\n"
        latex += "    \\hline\n"
        
        for lang in ['code-mixed', 'english', 'hindi', 'other']:
            if lang in lang_dist:
                count = lang_dist[lang]
                pct = (count / total) * 100
                lang_label = lang.replace('-', ' ').title()
                latex += f"    {lang_label} & {count} & {pct:.2f} \\\\\n"
        
        latex += "    \\hline\n"
        latex += f"    \\textbf{{Total}} & {total} & 100.00 \\\\\n"
        latex += "    \\hline\n"
        latex += "    \\end{tabular}\n"
        latex += "    \\label{tab:language_composition}\n"
        latex += "\\end{table}"
        
        import os
        os.makedirs('paper_tables', exist_ok=True)
        
        with open('paper_tables/table_language_composition.tex', 'w') as f:
            f.write(latex)
        
        print("\n✓ Saved: paper_tables/table_language_composition.tex")
        
        # Also save as CSV
        summary_data = []
        for lang in lang_dist.index:
            count = lang_dist[lang]
            pct = (count / total) * 100
            summary_data.append({
                'Language Type': lang,
                'Count': count,
                'Percentage': f"{pct:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('paper_tables/language_composition.csv', index=False)
        print("✓ Saved: paper_tables/language_composition.csv")
        
        return latex

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('paper_figures', exist_ok=True)
    os.makedirs('paper_tables', exist_ok=True)
    
    print("="*70)
    print("EXPERIMENT 1: LANGUAGE DETECTION & CODE-MIXING ANALYSIS")
    print("="*70)
    
    analyzer = LanguageMixingAnalysis()
    
    # Analyze dataset
    analyzer.analyze_dataset()
    
    # By athlete
    analyzer.analyze_by_athlete()
    
    # Show samples
    analyzer.sample_tweets_by_type()
    
    # Visualize
    analyzer.plot_language_distribution()
    
    # Create table
    analyzer.create_summary_table()
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT 1 COMPLETE!")
    print("="*70)
    print("\nKey Findings:")
    print("- Quantified code-mixing prevalence in dataset")
    print("- Generated publication-quality figure")
    print("- Created LaTeX table for paper")