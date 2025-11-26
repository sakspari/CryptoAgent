import os
from dotenv import load_dotenv
from agno.agent import Agent
from src.agents.news_agent import news_agent
from src.agents.crypto_agent import crypto_agent
from src.agents.telegram_agent import telegram_agent

# Load environment variables
load_dotenv()

def main():
    print("Starting Crypto Agent Pipeline...")
    
    # 1. Get Prediction (First, to know which asset is the best pick)
    print("Generating Prediction...")
    # We ask the agent to find the best pick. The tool does the heavy lifting.
    prediction_response = crypto_agent.run("Find the best crypto asset to buy for tomorrow and provide SL/TP in IDR.")
    prediction_result = prediction_response.content
    print(f"Prediction Result: {prediction_result}")

    # Extract the asset name from the prediction result to search for specific news
    # Simple heuristic: Check which asset is mentioned as "Best Pick"
    # Or just search for general crypto news if parsing is hard.
    # Let's try to extract or just search for the top assets.
    target_asset = "Crypto Market"
    if "BTC" in prediction_result: target_asset = "Bitcoin"
    elif "ETH" in prediction_result: target_asset = "Ethereum"
    elif "XRP" in prediction_result: target_asset = "XRP"
    elif "BNB" in prediction_result: target_asset = "BNB"
    
    # 2. Get News for the target asset
    print(f"Fetching News for {target_asset}...")
    news_response = news_agent.run(f"Get the latest news for {target_asset}. Summarize it briefly.")
    news_summary = news_response.content
    
    # 3. Construct Message
    final_message = f"""
🚀 **Crypto Agent Daily Report**

{prediction_result}

📰 **News Insight ({target_asset})**:
{news_summary}

*Disclaimer: Automated prediction. DYOR.*
    """

    # 4. Send to Telegram
    print("Sending to Telegram...")
    telegram_response = telegram_agent.run(f"Send this message to Telegram:\n\n{final_message}")
    print(f"Telegram Status: {telegram_response.content}")

if __name__ == "__main__":
    main()
