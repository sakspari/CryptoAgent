from agno.agent import Agent
from agno.models.google import Gemini
from duckduckgo_search import DDGS
import os

def get_crypto_news(symbol: str, max_results: int = 5) -> str:
    """
    Searches for the latest news about a crypto symbol.
    
    Args:
        symbol (str): The crypto symbol (e.g., "BTC", "Bitcoin").
        max_results (int): Number of results to return.
        
    Returns:
        str: A formatted string containing news titles and snippets.
    """
    try:
        results = DDGS().text(f"{symbol} crypto news", max_results=max_results)
        news_summary = ""
        for i, res in enumerate(results):
            news_summary += f"{i+1}. {res['title']}: {res['body']} ({res['href']})\n"
        return news_summary
    except Exception as e:
        return f"Error fetching news: {e}"

# Create the News Agent
news_agent = Agent(
    name="News Agent",
    role="Crypto News Reporter",
    instructions="You are a crypto news reporter. Your goal is to find the latest and most relevant news for a given cryptocurrency.",
    tools=[get_crypto_news],
    model=Gemini(id="gemini-flash-latest"),
    markdown=True,
)
