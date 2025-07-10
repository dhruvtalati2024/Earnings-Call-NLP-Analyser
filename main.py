import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

with open('AAPL.txt', 'r', encoding='utf-8') as file:
    transcript = file.read()

def run_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def run_finbert(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ['negative', 'neutral', 'positive']
    sent = {label: float(probs[0][i]) for i, label in enumerate(labels)}
    sent['pos_minus_neg'] = sent['positive'] - sent['negative']
    return sent

def get_stock_data(ticker, earnings_date, window=7):
    ed = pd.to_datetime(earnings_date)
    df = yf.download(ticker, start=ed - pd.Timedelta(days=window), end=ed + pd.Timedelta(days=window), auto_adjust=False)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    df['Return'] = df['Adj Close'].pct_change()
    return df

def plot_stock_data(df, earnings_date):
    ed = pd.to_datetime(earnings_date)
    plt.figure(figsize=(12,6))
    sns.lineplot(x=df.index, y=df['Adj Close'].values.flatten(), marker='o')
    plt.axvline(ed, color='red', linestyle='--', label='Earnings Date')
    plt.title('AAPL Stock Price Around Earnings (May 1, 2025)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    sns.barplot(x=df.index.strftime('%Y-%m-%d'), y=df['Return'].values.flatten(), color='skyblue')
    plt.axvline(len(df)//2, color='red', linestyle='--', label='Earnings Date')
    plt.xticks(rotation=45)
    plt.title('AAPL Daily Returns Around Earnings')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

earnings_date = '2025-05-01'
vader = run_vader(transcript)
print("\nVADER:", vader)
finbert = run_finbert(transcript)
print("\nFinBERT:", finbert)

data = get_stock_data('AAPL', earnings_date)
print("\nStock Data:\n", data[['Adj Close','Return']])

pre = data[data.index < pd.to_datetime(earnings_date)]['Return'].mean()
post = data[data.index >= pd.to_datetime(earnings_date)]['Return'].mean()
print(f"\nAvg Pre-Earnings Return: {pre:.4f}")
print(f"Avg Post-Earnings Return: {post:.4f}")

plot_stock_data(data, earnings_date)

print("\nSummary")
print(f"VADER Pos: {vader['pos']:.2f}, Neu: {vader['neu']:.2f}, Neg: {vader['neg']:.2f}, Compound: {vader['compound']:.2f}")
print(f"FinBERT Pos: {finbert['positive']:.2f}, Neu: {finbert['neutral']:.2f}, Neg: {finbert['negative']:.2f}, Pos-Neg: {finbert['pos_minus_neg']:.2f}")
print(f"Returns Pre: {pre:.4f}, Post: {post:.4f}")
