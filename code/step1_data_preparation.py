
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import random

# Set random seed
random.seed(42)
np.random.seed(42)

class TweetDataLoader:
    def __init__(self, json_folder, translated_folder=None):
        """
        json_folder: path to folder with original JSON files
        translated_folder: path to folder with translated JSON files (optional)
        """
        self.json_folder = json_folder
        self.translated_folder = translated_folder
        self.athletes = [
            'viratkohli',
            'harmanpreetkaur', 
            'sakshimalik',
            'saritadevi',
            'sushilkumarwrestler',
            'vijendersingh'
        ]
        self.all_tweets = []
        
    def load_all_tweets(self):
        """Load all athlete tweets into one dataframe"""
        all_data = []
        
        for athlete in self.athletes:
            # Load original tweets
            original_file = os.path.join(self.json_folder, f"{athlete}.json")
            
            if os.path.exists(original_file):
                print(f"Loading {athlete}...")
                with open(original_file, 'r', encoding='utf-8') as f:
                    tweets = json.load(f)
                
                # Add athlete name to each tweet
                for tweet in tweets:
                    tweet['athlete_name'] = athlete
                    tweet['athlete_display_name'] = self._format_athlete_name(athlete)
                
                all_data.extend(tweets)
                print(f"  ✓ Loaded {len(tweets)} tweets for {athlete}")
            else:
                print(f"  ✗ File not found: {original_file}")
        
        self.all_tweets = pd.DataFrame(all_data)
        
        # Fix column names - remove spaces and standardize
        self.all_tweets.columns = self.all_tweets.columns.str.replace(' ', '_')
        
        # Now rename to standard names
        column_mapping = {
            'tweet_id': 'tweet_id',
            'tweet_user_username': 'username',
            'tweet_rawcontent': 'raw_content'
        }
        
        self.all_tweets.rename(columns=column_mapping, inplace=True)
        
        print(f"\n✓ Total tweets loaded: {len(self.all_tweets)}")
        print(f"✓ Columns: {self.all_tweets.columns.tolist()}")
        
        return self.all_tweets
    
    def load_translated_tweets(self):
        """Load translated versions if available"""
        if not self.translated_folder or not os.path.exists(self.translated_folder):
            print("\n⚠ No translated folder provided or folder doesn't exist")
            return self.all_tweets
        
        translated_data = {}
        
        for athlete in self.athletes:
            # Try multiple possible filename patterns
            possible_files = [
                f"{athlete}_translated.json",
                f"{athlete}_eng.json",
                f"{athlete}.json"
            ]
            
            for filename in possible_files:
                trans_file = os.path.join(self.translated_folder, filename)
                
                if os.path.exists(trans_file):
                    print(f"  Loading translations from {filename}...")
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        tweets = json.load(f)
                    
                    # Create mapping of tweet_id to translated content
                    for tweet in tweets:
                        # Handle both space and underscore in key names
                        tweet_id = tweet.get('tweet_id') or tweet.get('tweet id')
                        # Look for various possible keys for translated content
                        translated = (tweet.get('translated_content') or 
                                    tweet.get('tweet_translated') or
                                    tweet.get('tweet_rawcontent') or
                                    tweet.get('tweet rawcontent'))
                        if tweet_id and translated:
                            translated_data[tweet_id] = translated
                    break
        
        if translated_data:
            self.all_tweets['translated_content'] = self.all_tweets['tweet_id'].map(translated_data)
            print(f"✓ Loaded translations for {len(translated_data)} tweets")
            
            # Fill missing translations with raw content
            n_missing = self.all_tweets['translated_content'].isna().sum()
            if n_missing > 0:
                print(f"  ⚠ {n_missing} tweets missing translations - using raw content")
                self.all_tweets['translated_content'].fillna(self.all_tweets['raw_content'], inplace=True)
        else:
            print("⚠ No translations found - using raw_content as translated_content")
            self.all_tweets['translated_content'] = self.all_tweets['raw_content']
        
        return self.all_tweets
    
    def _format_athlete_name(self, athlete_key):
        """Convert athlete key to display name"""
        name_map = {
            'viratkohli': 'Virat Kohli',
            'harmanpreetkaur': 'Harmanpreet Kaur',
            'sakshimalik': 'Sakshi Malik',
            'saritadevi': 'Sarita Devi',
            'sushilkumarwrestler': 'Sushil Kumar',
            'vijendersingh': 'Vijender Singh'
        }
        return name_map.get(athlete_key, athlete_key)
    
    def basic_preprocessing(self):
        """Clean and preprocess tweets"""
        print("\nPreprocessing tweets...")
        
        # Remove duplicates
        before = len(self.all_tweets)
        self.all_tweets.drop_duplicates(subset=['tweet_id'], inplace=True)
        print(f"  Removed {before - len(self.all_tweets)} duplicate tweets")
        
        # Remove empty tweets
        self.all_tweets = self.all_tweets[self.all_tweets['raw_content'].notna()]
        self.all_tweets = self.all_tweets[self.all_tweets['raw_content'].str.strip() != '']
        
        # Remove tweets that are just URLs or very short
        self.all_tweets = self.all_tweets[self.all_tweets['raw_content'].str.len() > 10]
        
        print(f"  ✓ Final tweet count: {len(self.all_tweets)}")
        
        return self.all_tweets
    
    def create_stratified_sample(self, n_total=500):
        """Create stratified sample for annotation"""
        samples_per_athlete = n_total // 6
        
        sampled_tweets = []
        
        for athlete in self.all_tweets['athlete_name'].unique():
            athlete_tweets = self.all_tweets[self.all_tweets['athlete_name'] == athlete]
            n_sample = min(samples_per_athlete, len(athlete_tweets))
            sample = athlete_tweets.sample(n=n_sample, random_state=42)
            sampled_tweets.append(sample)
            print(f"  Sampled {n_sample} tweets from {self._format_athlete_name(athlete)}")
        
        sample_df = pd.concat(sampled_tweets, ignore_index=True)
        
        # Shuffle the combined sample
        sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✓ Created stratified sample: {len(sample_df)} tweets")
        
        return sample_df
    
    def export_for_annotation(self, sample_df, output_file='tweets_for_annotation.csv'):
        """Export sample for human annotation"""
        
        # Create annotation template
        annotation_df = sample_df[['tweet_id', 'athlete_display_name', 'raw_content']].copy()
        
        # Add translated content if available
        if 'translated_content' in sample_df.columns:
            annotation_df['translated_content'] = sample_df['translated_content']
        
        # Add empty annotation columns
        annotation_df['annotator1_sentiment'] = ''
        annotation_df['annotator2_sentiment'] = ''
        annotation_df['annotator3_sentiment'] = ''
        annotation_df['final_sentiment'] = ''  # For resolved/gold label
        annotation_df['notes'] = ''
        
        # Reorder columns
        cols = ['tweet_id', 'athlete_display_name', 'raw_content']
        if 'translated_content' in annotation_df.columns:
            cols.append('translated_content')
        cols.extend(['annotator1_sentiment', 'annotator2_sentiment', 
                     'annotator3_sentiment', 'final_sentiment', 'notes'])
        
        annotation_df = annotation_df[cols]
        
        # Export
        annotation_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n{'='*70}")
        print(f"✓ EXPORTED: {output_file}")
        print(f"{'='*70}")
        print(f"\nAnnotation Instructions:")
        print("-" * 70)
        print("1. For each tweet, label the sentiment as:")
        print("   • positive  (supportive, praising, encouraging)")
        print("   • neutral   (factual, informational, no clear sentiment)")
        print("   • negative  (critical, mocking, hateful, cyberbullying)")
        print("\n2. Fill in columns: annotator1_sentiment, annotator2_sentiment")
        print("   (Get at least 2 people to annotate)")
        print("\n3. After discussion, fill 'final_sentiment' with agreed label")
        print("\n4. Use 'notes' for difficult cases or ambiguous tweets")
        print("-" * 70)
        
        return output_file

# ============================================================================
# MAIN EXECUTION - STEP 1
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STEP 1: DATA PREPARATION FOR BASELINE EXPERIMENTS")
    print("="*70)
    
    # These paths match your folder structure
    JSON_FOLDER = "json_files"  
    TRANSLATED_FOLDER = "translated_json_files"  
    
    # Initialize loader
    loader = TweetDataLoader(JSON_FOLDER, TRANSLATED_FOLDER)
    
    # Step 1a: Load all tweets
    print("\n[1/5] Loading all athlete tweets...")
    all_tweets_df = loader.load_all_tweets()
    
    # Step 1b: Load translated versions (if available)
    print("\n[2/5] Loading translated tweets...")
    all_tweets_df = loader.load_translated_tweets()
    
    # Step 1c: Basic preprocessing
    print("\n[3/5] Preprocessing tweets...")
    all_tweets_df = loader.basic_preprocessing()
    
    # Step 1d: Create stratified sample
    print("\n[4/5] Creating stratified sample for annotation...")
    sample_df = loader.create_stratified_sample(n_total=500)
    
    # Step 1e: Export for annotation
    print("\n[5/5] Exporting annotation file...")
    output_file = loader.export_for_annotation(sample_df)
    
    # Save full dataset as well
    all_tweets_df.to_csv('all_tweets_combined.csv', index=False, encoding='utf-8')
    print(f"\n✓ Also saved complete dataset: all_tweets_combined.csv")
    
    # Statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total tweets: {len(all_tweets_df)}")
    print(f"\nTweets per athlete:")
    print(all_tweets_df['athlete_display_name'].value_counts())
    print(f"\n✓ Ready for annotation!")
    print(f"{'='*70}")