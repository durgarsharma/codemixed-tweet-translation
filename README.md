# Code-Mixed Sentiment Analysis: Evaluating LLM Translation Against Multilingual Transformers on Twitter Data

> https://durgarsharma.github.io/codemixed-tweet-translation/

## Abstract
Hindi and English (Hi–En) code mixing is widespread across Indian social media and remains a major challenge for sentiment analysis and cyberbullying detection. Current multilingual models struggle with this type of data, often giving inconsistent predictions and showing lower confidence. In this work, 27,314 tweets directed at six Indian athletes across three sports (2013–2023) to understand how code mixing affects automated sentiment classification. It was found that 12\% of tweets contain Hi–En code mixing, with significant differences across athletes, and that 48.4\% of all tweets are negative, with Wrestling athletes receiving the highest levels of negativity. Compared the state of the art multilingual models (XLM-RoBERTa, mBERT) with an LLM based translation pipeline and found that these models show only fair agreement on code mixed tweets, highlighting the difficulty of the task. Human evaluation on 498 tweets provides gold standard labels and shows that both multilingual models reach only moderate performance. Overall, the results show the limitations of current multilingual systems for processing code mixed text and demonstrate that translation based workflows offer a strong alternative for sentiment analysis in linguistically diverse social media settings.

## Result 1: Multiligual Models Disagree
- Low agreement: Cohen's κ = 0.253 between XLM-RoBERTa and mBERT
- Statistical significance: χ² = 11,761, p < 0.001
- Extreme divergence: XLM-RoBERTa classifies 42.9%, and mBERT classifies 3.8% as neutral
- Reduced confidence: Both models show significantly lower confidence on code-mixed text (p < 0.001)

## Result 2: Human Evaluation Validates Difficulty
- Moderate inter annotator agreement: Cohen's κ = 0.586
- Both models achieve F1 ≈ 0.59 against gold standard
- Neutral failure: mBERT F1 = 0.07 for neutral sentiment

## Result 3: Translation Works Well
- 76.4% sentiment preservation between raw and translated
- Systematic improvements: 11.5% neutral → negative (better detection)
- Minimal distortion: Only 13.3 character average length increases



## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
