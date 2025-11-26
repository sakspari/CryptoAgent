from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
import feedparser

def get_crypto_news(query: str = "latest") -> str:
    """
    Fetches the latest crypto news from CoinDesk RSS feed.
    
    Args:
        query (str): Not used for RSS, but kept for compatibility.
        
    Returns:
        str: A summary of the latest news from CoinDesk.
    """
    import random
    import re
    
    # List of high-quality RSS feeds
    rss_feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
        "https://finance.yahoo.com/news/rssindex",
        "https://decrypt.co/feed",
        "https://thedefiant.io/feed"
    ]
    
    try:
        # Select fewer feeds to save tokens (3 instead of 4)
        selected_feeds = random.sample(rss_feeds, min(3, len(rss_feeds)))
        news_summary = ""
        
        for url in selected_feeds:
            try:
                feed = feedparser.parse(url)
                if not feed.entries: continue
                
                source_name = "Unknown"
                if "coindesk" in url: source_name = "CoinDesk"
                elif "cointelegraph" in url: source_name = "CoinTelegraph"
                elif "cryptoslate" in url: source_name = "CryptoSlate"
                elif "yahoo" in url: source_name = "Yahoo"
                elif "decrypt" in url: source_name = "Decrypt"
                elif "thedefiant" in url: source_name = "Defiant"
                
                # Get top 2 entries per feed (was 3)
                for entry in feed.entries[:2]:
                    title = entry.get('title', 'No Title')
                    summary = entry.get('summary', 'No Summary')
                    
                    # Clean HTML
                    summary = re.sub('<[^<]+?>', '', summary)
                    # Remove newlines and extra spaces
                    summary = " ".join(summary.split())
                    
                    # Truncate aggressively (120 chars)
                    if len(summary) > 120:
                        summary = summary[:120] + "..."
                        
                    # Compact format
                    news_summary += f"- [{source_name}] {title}: {summary}\n"
            except:
                continue
                
        return news_summary if news_summary else "No news found."
    except Exception as e:
        return f"Error fetching RSS news: {e}"

import os

# Create the News Agent
news_agent = Agent(
    name="News Agent",
    instructions="""
    You are a crypto trend spotter. Your goal is to identify ONE cryptocurrency with the most SIGNIFICANT market-moving news for tomorrow.
    
    1. Use the `get_crypto_news` tool to get the latest headlines.
    2. Analyze the news to pick the single asset with the strongest catalyst (Bullish OR Bearish).
       - Do NOT just pick Bitcoin unless there is specific, significant news about it.
       - Look for altcoins with major partnerships, hacks, regulatory news, or upgrades.
       - If the news is negative (e.g., a hack, ban, or lawsuit), pick that asset as a potential SHORT candidate.
    3. Return the response in the following strict format:
       TICKER: [Yahoo Finance Symbol, e.g. BTC-USD, ETH-USD, SOL-USD]
       REASON: [A brief 1-2 sentence summary of the specific news/catalyst. Mention if it is Bullish or Bearish news.]
    """,
    tools=[get_crypto_news],
    model=Gemini(id=os.getenv("GEMINI_MODEL_ID", "gemini-flash-latest")),
    retries=int(os.getenv("AGENT_RETRIES", 3)),
    delay_between_retries=int(os.getenv("RETRY_DELAY", 5)),
    markdown=True,
)
