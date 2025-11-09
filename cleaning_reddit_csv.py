import pandas as pd
import re
from textblob import TextBlob

# Read the CSV file
df = pd.read_csv('scraping/tmobile-reddit.csv')

print(f"Initial shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================
# DATA CLEANING STEPS
# ============================================

# 1. Remove duplicates based on title and text
print("\n--- Removing duplicates ---")
initial_count = len(df)
df = df.drop_duplicates(subset=['title', 'text'], keep='first')
print(f"Removed {initial_count - len(df)} duplicate rows")

# 2. Handle missing values
print("\n--- Handling missing values ---")
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# Fill empty text with title if text is missing
df['text'] = df['text'].fillna(df['title'])

# Remove rows where both title and text are missing
df = df.dropna(subset=['title', 'text'], how='all')

print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# 3. Clean text data
print("\n--- Cleaning text data ---")

def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*|__|\[|\]|\(|\)', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

df['title'] = df['title'].apply(clean_text)
df['text'] = df['text'].apply(clean_text)

# 4. Create combined feedback text column
print("\n--- Creating FeedbackText column ---")
df['FeedbackText'] = df['title'] + '. ' + df['text']
df['FeedbackText'] = df['FeedbackText'].str.strip('. ')

# 5. Remove very short entries (less than 10 characters)
print("\n--- Removing very short entries ---")
initial_count = len(df)
df = df[df['FeedbackText'].str.len() >= 10]
print(f"Removed {initial_count - len(df)} rows with very short text")

# 6. Remove non-English posts (optional - simple heuristic)
print("\n--- Filtering non-English posts ---")
def is_likely_english(text):
    """Simple heuristic to detect English text"""
    if not text or len(text) < 20:
        return True
    
    # Check for common English words
    english_words = ['the', 'is', 'are', 'have', 'with', 'for', 'from', 'this', 'that']
    text_lower = text.lower()
    word_count = sum(1 for word in english_words if word in text_lower)
    
    return word_count >= 2

initial_count = len(df)
df = df[df['FeedbackText'].apply(is_likely_english)]
print(f"Removed {initial_count - len(df)} likely non-English rows")

# 7. Reset index
df = df.reset_index(drop=True)

print(f"\n--- Final cleaned data shape: {df.shape} ---")

# ============================================
# ADD SENTIMENT ANALYSIS
# ============================================

print("\n--- Adding sentiment analysis ---")

def get_sentiment_score(text):
    """
    Calculate sentiment score using TextBlob
    Returns: 0 (negative), 0.5 (neutral), 1 (positive)
    """
    if not text or len(text) < 5:
        return 0.5
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        # Map polarity (-1 to 1) to our scale (0, 0.5, 1)
        if polarity < -0.1:
            return 0  # Negative
        elif polarity > 0.1:
            return 1  # Positive
        else:
            return 0.5  # Neutral
    except:
        return 0.5

# Apply sentiment analysis
df['Sentiment'] = df['FeedbackText'].apply(get_sentiment_score)

# Add sentiment label for easier interpretation
def get_sentiment_label(score):
    if score == 0:
        return 'Negative'
    elif score == 0.5:
        return 'Neutral'
    else:
        return 'Positive'

df['SentimentLabel'] = df['Sentiment'].apply(get_sentiment_label)

# ============================================
# DISPLAY STATISTICS
# ============================================

print("\n--- Sentiment Distribution ---")
print(df['SentimentLabel'].value_counts())
print("\nPercentage distribution:")
print(df['SentimentLabel'].value_counts(normalize=True) * 100)

# ============================================
# SAVE CLEANED DATA
# ============================================

# Select final columns
final_columns = ['title', 'text', 'FeedbackText', 'Sentiment', 'SentimentLabel']
df_final = df[final_columns]

# Save to CSV
output_file = 'tmobile_reddit_cleaned.csv'
df_final.to_csv(output_file, index=False)
print(f"\n✓ Cleaned data saved to: {output_file}")

# Display sample of final data
print("\n--- Sample of cleaned data ---")
print(df_final.head(10))

# Show some examples from each sentiment category
print("\n--- Examples by Sentiment ---")
for sentiment in ['Negative', 'Neutral', 'Positive']:
    print(f"\n{sentiment} Example:")
    sample = df_final[df_final['SentimentLabel'] == sentiment].head(1)
    if not sample.empty:
        print(f"Title: {sample.iloc[0]['title'][:80]}...")
        print(f"Text: {sample.iloc[0]['FeedbackText'][:150]}...")