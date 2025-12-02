# T-Emotions: T-Mobile Customer Happiness Index

Turn customer feedback into actionable insights in real-time—detect issues faster, celebrate wins, and boost customer satisfaction with AI-powered sentiment analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Data Collection](#data-collection)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Modeling](#modeling)
- [Dashboard](#dashboard)
- [Key Insights](#key-insights)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

T-Emotions is an AI-powered system designed to analyze customer feedback in real-time. By leveraging sentiment analysis, it identifies issues, celebrates wins, and generates actionable insights for businesses.

This project focuses on T-Mobile's Reddit feedback, but the system can be adapted for any company with customer reviews or social media mentions.

---

## Problem Statement

Businesses receive hundreds of customer feedback posts daily. Identifying urgent issues or positive moments manually is time-consuming, which delays responses and may result in customer churn.

The goal of T-Emotions is to **automatically detect sentiment**, categorize feedback, and provide insights instantly.

---

## Solution

We built an end-to-end AI pipeline that:

1. Scrapes customer feedback from Reddit using Python and PRAW.
2. Cleans and preprocesses the text data to remove noise and duplicates.
3. Labels sentiment using TextBlob and trains machine learning models (Logistic Regression, Random Forest).
4. Deploys the best-performing model in a **Streamlit dashboard** for live monitoring.

---

## Features

- **Dashboard Overview**: Displays a weighted "Happiness Index," live metrics, and instant alerts.
- **Historical Analytics**: Visualize sentiment trends, keyword analysis, and customer insights over time.
- **Live Sentiment Detector**: Paste any comment to get instant positive/neutral/negative predictions.
- **Issue Detection**: Categorizes feedback into Network, Billing, Customer Service, or Device issues.
- **Moments of Delight**: Highlights the best customer testimonials for team celebrations or marketing.

---

## Data Collection

- **Source**: Reddit T-Mobile community posts
- **Number of posts**: 100 (86 high-quality after cleaning)
- **Tools**: Python, PRAW (Reddit API)

---

## Data Cleaning & Preprocessing

- Removed duplicates, URLs, emojis, and markdown formatting.
- Filtered out posts that were too short or not in English.
- 7-stage cleaning pipeline to ensure high-quality data.

---

## Modeling

- **Sentiment Labeling**: TextBlob
- **Models Used**: Logistic Regression, Random Forest
- **Model Selection**: Chose the model with the best accuracy and performance.
- **Serialization**: Saved trained model (`sentiment_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`) for deployment.

---

## Dashboard

- Built with **Streamlit** for interactive visualization.
- **Files**: `dashboard.py`
- **Key Features**:
  - Live sentiment detection
  - Historical analytics charts
  - Issue classification and alerts
  - Moments of delight

---

## Key Insights

From the analysis of 86 Reddit posts:

- **54.7%** of feedback was positive (e.g., praise for 5G speeds and pricing).
- Major issues include:
  - eSIM transfers
  - Billing errors
  - Inconsistent 5G UC performance

These insights demonstrate how quickly businesses can detect customer pain points and successes.

---

## Installation

Ensure you have Python 3.9+ installed. Then install dependencies using pip:
```bash
pip install pandas==2.1.0 numpy==1.26.0 scikit-learn==1.3.0 textblob==0.17.1 streamlit==1.25.0 praw==7.7.0 matplotlib==3.8.1 seaborn==0.12.3 plotly==5.16.0
```

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/mahashanawaz/tmobile-happiness-index.git
cd tmobile-happiness-index
```

2. Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

3. Explore the dashboard for sentiment insights, issue detection, and historical analysis.

---

## Future Work

- Implement TF-IDF-based keyword extraction with bigram/trigram support.
- Connect to live Reddit and Twitter feeds for real-time monitoring.
- Expand to multiple carriers and platforms.
- Predict churn risk based on sentiment trends.

---

## Technologies Used

- **Languages**: Python
- **Libraries**: pandas, numpy, scikit-learn, textblob, Streamlit, PRAW, matplotlib, seaborn, plotly
- **Data Storage**: CSV, Pickle
- **Visualization**: Streamlit dashboard, plots, heatmaps

---

## Repository Structure
```
tmobile-happiness-index/
├── scraping/
├── cleaning_reddit_csv.py
├── train_sentiment_model.py
├── dashboard.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── tmobile_reddit_cleaned.csv
├── confusion_matrix.png
└── README.md
```

---

## Acknowledgements

- Reddit API (PRAW)
- HackUTD 2025
- Open-source ML and NLP libraries
- Python community tutorials and documentation
