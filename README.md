# Do Multilingual Transformers Beat Translation? Sentiment Analysis on Hindi and English Code Mixed Sports Discourse

## Abstract
Hindi and English (Hi-En) code mixing is common on Indian social media, yet sentiment analysis on such text is difficult to evaluate without a human annotated gold standard. We compare two approaches to code mixed sentiment classification, multilingual transformers (XLM-RoBERTa, mBERT) and an LLM based translation pipeline that translates text to English before classifying it with VADER. Using 27,314 tweets directed at six Indian athletes and a 498 tweet gold standard annotated by three bilingual raters (Fleiss’ κ = 0.615, Krippendorff’s α = 0.616), we observe that the translation pipeline (weighted F1 = 0.589) performs comparably to XLM-RoBERTa (McNemar p=0.598) and approaches mBERT (F1 = 0.590, p = 0.046). Error analysis reveals distinct failure modes, XLM-RoBERTa over predicts neutral for polar tweets while mBERT over predicts negative, yet all three approaches converge at the same F1 ceiling, suggesting the bottleneck lies in semantic ambiguity rather than code mixing itself. For Hi-En sentiment analysis, this suggests translation pipelines could serve as a light weight alternative to multilingual transformers.

## Key Findings

### Finding 1: All Approaches Hit the Same Ceiling
- Translation+VADER (F1 = 0.589), XLM-RoBERTa (F1 = 0.591), and mBERT (F1 = 0.590) converge at F1 ≈ 0.59
- McNemar's test: no significant difference between translation pipeline and XLM-RoBERTa (p = 0.598)
- Computationally cheaper translation pipelines match multilingual transformers

### Finding 2: Complementary Failure Modes
- XLM-RoBERTa over-predicts neutral: 88 positive and 84 negative tweets misclassified as neutral
- mBERT nearly ignores neutral (F1_neu = 0.075), instead over-predicting negative
- 65.1% of errors show no obvious surface-level trigger — failures are semantic, not linguistic

### Finding 3: Rigorous Human Evaluation
- Three bilingual annotators, 498 tweets
- Fleiss' κ = 0.615, Krippendorff's α = 0.616 (substantial agreement)
- Gold label distribution: 48.4% negative, 39.6% positive, 12.0% neutral

### Finding 4: Translation Preserves Sentiment
- 76.4% sentiment label retention after LLM translation
- Translation surfaces hostile content: 11.5% neutral → negative shift
- Translated VADER significantly outperforms raw VADER (McNemar p < 0.001)

### Finding 5: Sport-Specific Negativity
- Wrestling: 49.8% (XLM) to 76.6% (mBERT) negative
- Boxing: 38.9% (XLM) to 75.6% (mBERT) negative
- Cricket: 18.9% (XLM) to 44.8% (mBERT) negative

## Dataset
27,314 tweets directed at six Indian athletes across three sports (2013–2023).

| Athlete | Sport | Period | CM% | Tweets |
| :--- | :--- | :--- | :--- | :--- |
| Virat Kohli | Cricket | May–Oct 2021 | 2.3% | 4,892 |
| Harmanpreet Kaur | Cricket | May–Oct 2017 | 4.2% | 4,156 |
| Vijender Singh | Boxing | Feb–Jul 2013 | 0.5% | 4,234 |
| Sarita Devi | Boxing | Sep 2014–Feb 2015 | 6.8% | 4,421 |
| Sushil Kumar | Wrestling | May–Oct 2021 | 15.7% | 4,567 |
| Sakshi Malik | Wrestling | Jan–Jun 2023 | 48.1% | 5,044 |

CM% = code-mixed tweet proportion. Language composition: 87.84% English, 12.01% Hi-En code-mixed, 0.11% Hindi.

## Models
| Model | Role | Input |
| :--- | :--- | :--- |
| XLM-RoBERTa | Sentiment classification | Raw code-mixed tweets |
| mBERT | Sentiment classification | Raw code-mixed tweets |
| GPT-4o | Translation to English | Raw code-mixed tweets |
| Claude 3.5 Sonnet | Translation to English | Raw code-mixed tweets |
| VADER | Sentiment classification | Translated English tweets |

## Example
**Code-mixed Hi-En:** "Yaar, kal ka match bohot intense tha, but Virat ne amazing performance di!"
**Translated:** "Man, yesterday's match was very intense, but Virat gave an amazing performance!"

**Code-switched Hi-En:** "I can't believe we lost the game, lekin Virat ne bohot achha khela."
**Translated:** "I can't believe we lost the game, but Virat played really well."

