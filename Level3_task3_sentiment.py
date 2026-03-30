# Codveda Internship - Level 3, Task 3: NLP Sentiment Analysis
# Dataset: Sentiment Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ── 1. LOAD DATASET ──────────────────────────────────────
df = pd.read_csv('3) Sentiment dataset.csv')
print("Shape:", df.shape)
print(df[['Text', 'Sentiment']].head())
print("\nSentiment Distribution:\n", df['Sentiment'].value_counts())

# ── 2. CLEAN THE DATA ─────────────────────────────────────
df['Text'] = df['Text'].astype(str).str.strip()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
print("\nAfter cleaning:")
print(df['Sentiment'].value_counts())

# ── 3. TEXT PREPROCESSING ─────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters, numbers, punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['Text'].apply(preprocess_text)
print("\nSample cleaned text:")
print(df[['Text', 'cleaned_text']].head(3))

# ── 4. SENTIMENT ANALYSIS WITH TEXTBLOB ───────────────────
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['predicted_sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)
print("\nPredicted Sentiment Distribution:\n", df['predicted_sentiment'].value_counts())

# ── 5. COMPARE ACTUAL VS PREDICTED ────────────────────────
print("\nActual vs Predicted Sample:")
print(df[['Text', 'Sentiment', 'predicted_sentiment']].head(10))

# ── 6. PLOT 1: Actual Sentiment Distribution ──────────────
plt.figure(figsize=(7, 5))
colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
sentiment_counts = df['Sentiment'].value_counts()
plt.bar(sentiment_counts.index,
        sentiment_counts.values,
        color=[colors.get(s, '#95a5a6') for s in sentiment_counts.index])
plt.title('Actual Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150)
plt.show()

# ── 7. PLOT 2: TextBlob Predicted Distribution ────────────
plt.figure(figsize=(7, 5))
pred_counts = df['predicted_sentiment'].value_counts()
plt.bar(pred_counts.index,
        pred_counts.values,
        color=[colors.get(s, '#95a5a6') for s in pred_counts.index])
plt.title('TextBlob Predicted Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('predicted_distribution.png', dpi=150)
plt.show()

# ── 8. PLOT 3: Word Cloud (All Text) ──────────────────────
all_text = ' '.join(df['cleaned_text'].values)
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      colormap='viridis',
                      max_words=100).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words - All Sentiments')
plt.tight_layout()
plt.savefig('wordcloud_all.png', dpi=150)
plt.show()

# ── 9. PLOT 4: Word Cloud (Positive only) ─────────────────
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['cleaned_text'].values)
wc_pos = WordCloud(width=800, height=400,
                   background_color='white',
                   colormap='Greens',
                   max_words=100).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words - Positive Sentiment')
plt.tight_layout()
plt.savefig('wordcloud_positive.png', dpi=150)
plt.show()

# ── 10. PLOT 5: Word Cloud (Negative only) ────────────────
negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['cleaned_text'].values)
wc_neg = WordCloud(width=800, height=400,
                   background_color='white',
                   colormap='Reds',
                   max_words=100).generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words - Negative Sentiment')
plt.tight_layout()
plt.savefig('wordcloud_negative.png', dpi=150)
plt.show()

print("\nAll done! 5 plots saved.")