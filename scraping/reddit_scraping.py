import praw
import pandas as pd

with open("reddit_scrape.txt", "r") as reddit_file:
    CLIENT_ID = next(reddit_file).strip()
    CLIENT_SECRET = next(reddit_file).strip()
    USERNAME = next(reddit_file).strip()
    PASSWORD = next(reddit_file).strip()

USER_AGENT = f"tmobile-scraper by u/{USERNAME}"

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,  
    username=USERNAME,
    password=PASSWORD,
    user_agent=USER_AGENT
)

query = "tmobile"
results = []
limit = 100

for submission in reddit.subreddit("all").search(query, limit=limit, sort="new"):
    results.append({
        #"id": submission.id,
        "title": submission.title,
        "text": submission.selftext,
        #"created_utc": submission.created_utc,
    })

data = pd.DataFrame(results)
data.to_csv("tmobile-reddit.csv", index=False, encoding="utf-8")