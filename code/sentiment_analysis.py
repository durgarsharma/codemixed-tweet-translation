!pip install textblob
!pip install langdetect
!pip install nltk

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from langdetect import detect
from matplotlib import pyplot as plt
import json

nltk.download('vader_lexicon')

def process_file(filename, color, label):
    with open(filename, 'r') as f:
        data = json.load(f)

    if filename == 'sakshimalik_anthropic.json':
        if all(tweet_dict['translated_content'] == "This input is already entirely in English, so no translation is needed." for tweet_dict in data):
            lines = [tweet_dict['tweet rawcontent'] for tweet_dict in data]
        else:
            lines = [tweet_dict['translated_content'] for tweet_dict in data]
    elif filename == 'sakshimalik_translated_openai.json':
        lines = [tweet_dict['translated_content'] for tweet_dict in data]

    sia = SentimentIntensityAnalyzer()
    scores_final = []

    for line in lines:
        try:
            language = detect(line)
            if language != 'en': 
                pass
            score = sia.polarity_scores(line)['compound']
            scores_final.append(score)
        except:
            scores_final.append(0) 

    running_avg = []
    cnt = 1
    sum = 0.0
    for sc in scores_final:
        sum += sc
        cur_avg = sum / cnt
        cnt += 1
        running_avg.append(cur_avg)

    plt.plot(running_avg, color=color, label=label)

    neutral_cnt = len([sc for sc in scores_final if sc == 0])
    pos_cnt = len([sc for sc in scores_final if sc > 0])
    neg_cnt = len([sc for sc in scores_final if sc < 0])

    print(f"{label} - Positive: {pos_cnt}, Neutral: {neutral_cnt}, Negative: {neg_cnt}")

plt.figure(figsize=(12, 6))

process_file('sakshimalik_translated_openai.json', color='cyan', label='OpenAI')
process_file('sakshimalik_anthropic.json', color='pink', label='Anthropic')

plt.title('Sakshi Malik')
plt.xlabel('Time')
plt.ylabel('Average Sentiment')
plt.legend()

plt.show()
