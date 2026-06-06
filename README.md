# 🛡️ Hate Speech & Toxic Content Classifier

An end-to-end NLP system that automatically classifies social media posts
into **Hate Speech**, **Offensive Language**, or **Neither** —
built to demonstrate automated content moderation using classical ML.

🔗 **Live App:** [https://toxic-classifier-xyz.streamlit.app/]  
📓 **Kaggle Notebook:** [https://www.kaggle.com/code/radiantbright/toxiscan-toxic-content-classifier]

---

## Results

| Model | Accuracy | Weighted F1 | Hate Speech Recall |
|---|---|---|---|
| Naive Bayes (baseline) | 86.3% | 83.8% | 8% |
| Logistic Regression | 85.4% | 86.9% | 62% |
| **Linear SVM (final)** | **89.9%** | **88.5%** | **16%** |

**Key finding:** Accuracy alone is misleading on this dataset.
A naive model predicting "Offensive" for everything would score 77% accuracy.
Weighted F1 and per-class recall are the correct evaluation metrics.

---

## Dataset

- **Source:** [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- **Size:** 24,780 tweets (after cleaning)
- **Classes:** Hate Speech (5.8%), Offensive Language (77.4%), Neither (16.8%)
- **Challenge:** Severe class imbalance

---

## Pipeline

Raw Tweets
↓
Text Cleaning (lowercase, remove URLs/@mentions/RT/punctuation/numbers)
↓
Stopword Removal + Porter Stemming
↓
TF-IDF Vectorisation (unigrams + bigrams, 5,184 features)
↓
Train/Test Split (80/20, stratified)
↓
Model Training (Naive Bayes → Logistic Regression → Linear SVM)
↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
↓
Explainability (LR Coefficients → top words per class)
↓
Streamlit Deployment

---

## Tech Stack

- **Language:** Python 3.11
- **ML:** Scikit-learn (TF-IDF, LinearSVC, CalibratedClassifierCV)
- **NLP:** NLTK (PorterStemmer, stopwords)
- **Visualisation:** Matplotlib, Seaborn, WordCloud
- **Deployment:** Streamlit Community Cloud

---

## Limitations

- Hate Speech recall is 16% — minority class detection is the core challenge
- PorterStemmer is aggressive; lemmatisation would improve edge cases
- Bag-of-words cannot capture negation context ("I don't hate anyone")
- Future work: fine-tuned BERT/RoBERTa for contextual understanding