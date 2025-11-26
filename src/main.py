import os
import time
from dotenv import load_dotenv
from agno.agent import Agent
from src.agents.news_agent import news_agent
from src.agents.telegram_agent import send_telegram_message
from src.train_model import train_and_predict
from src.utils.db_manager import DBManager

# Load environment variables
load_dotenv()

def main():
    print("Starting Dynamic Crypto Agent Pipeline...")
    
    # Initialize database
    db = DBManager()
    recent_picks = db.get_recent_picks(10)
    history_summary = db.get_history_summary(3)
    print(f"📜 {history_summary}")
    
    # 1. Get Trending Asset from News Agent
    print("Consulting News Agent for the best pick...")
    
    # Format recent picks for context
    recent_list = "\n".join([f"- {p['ticker']} ({p['direction']})" for p in recent_picks]) if recent_picks else "None"
    
    try:
        prompt = f"""Find the best crypto asset to buy for tomorrow.

IMPORTANT: Avoid these recently picked assets (aim for diversity):
{recent_list}

Pick a NEW asset with strong market-moving news."""
        
        news_response = news_agent.run(prompt)
        content = news_response.content.strip()
        
        # Debug: Show raw response
        print(f"\n🔍 News Agent Raw Response:\n{content}\n")
        
        # Parse TICKER and REASON (case-insensitive, handle markdown)
        ticker = None
        reason = None
        
        lines = content.split('\n')
        for line in lines:
            # Remove markdown formatting (**, *, etc.)
            clean_line = line.replace("**", "").replace("*", "").strip()
            line_upper = clean_line.upper()
            
            if line_upper.startswith("TICKER:"):
                ticker = clean_line.split(":", 1)[1].strip().upper()
            elif line_upper.startswith("REASON:"):
                reason = clean_line.split(":", 1)[1].strip()
        
        # Fallback to defaults if parsing failed
        if not ticker:
            print("⚠️ WARNING: Could not parse TICKER from response, using default BTC-USD")
            ticker = "BTC-USD"
        if not reason:
            print("⚠️ WARNING: Could not parse REASON from response")
            reason = "Market analysis."
        
        # Basic validation
        if not ticker.endswith("-USD"):
            if len(ticker) <= 5 and ticker.isalpha():
                ticker = f"{ticker}-USD"
        
        print(f"✅ Selected Asset: {ticker}")
        print(f"✅ Reason: {reason}")
        
    except Exception as e:
        print(f"❌ News Agent failed with error: {e}")
        print("Defaulting to BTC-USD due to News Agent failure.")
        ticker = "BTC-USD"
        reason = "Automated fallback selection due to News Agent error."

    # Throttle to prevent 429
    print("Sleeping for 20s to respect rate limits...")
    time.sleep(20)

    # 2. Dynamic Training & Prediction
    print(f"Training model and predicting for {ticker}...")
    result = train_and_predict(ticker)
    
    if "error" in result:
        error_msg = f"Analysis failed for {ticker}: {result['error']}"
        print(error_msg)
        send_telegram_message(f"⚠️ Error: {error_msg}")
        return

    # 3. Construct Message
    direction_emoji = "🚀" if result['direction'] == "BULLISH" else "🔻"
    
    if result['direction'] == "BULLISH":
        final_message = f"""
{direction_emoji} **Daily Crypto Pick: {result['ticker']}**

*   **Signal**: {result['direction']}
*   **Forecast**: {result['pred_pct']:.2f}% {result['direction'].lower()}
*   **Current Price**: Rp {result['current_price_idr']:,.0f} ($ {result['current_price_usd']:.2f})

🎯 **Trade Setup**:
*   **Buy Zone**: ~Rp {result['current_price_idr']:,.0f}
*   **Stop Loss**: Rp {result['sl_idr']:,.0f} ($ {result['sl_usd']:.2f})
*   **Take Profit**: Rp {result['tp_idr']:,.0f} ($ {result['tp_usd']:.2f})

📰 **Why this coin?**
{reason}

*Analysis based on live dynamic model training.*
*DYOR.*
        """
    elif result['direction'] == "BEARISH":
        final_message = f"""
{direction_emoji} **Daily Crypto Pick: {result['ticker']}**

*   **Signal**: {result['direction']}
*   **Forecast**: {result['pred_pct']:.2f}% {result['direction'].lower()}
*   **Current Price**: Rp {result['current_price_idr']:,.0f} ($ {result['current_price_usd']:.2f})

📉 **Market Outlook**:
The model predicts a price drop to **Rp {result['predicted_price_idr']:,.0f}** ($ {result['predicted_price_usd']:.2f}) tomorrow.
No long trade recommended at this time.

📰 **News Context**:
{reason}

*Analysis based on live dynamic model training.*
*DYOR.*
        """
    else: # NEUTRAL
        final_message = f"""
⚖️ **Daily Crypto Pick: {result['ticker']}**

*   **Signal**: {result['direction']}
*   **Forecast**: {result['pred_pct']:.2f}% (Sideways)
*   **Current Price**: Rp {result['current_price_idr']:,.0f} ($ {result['current_price_usd']:.2f})

😐 **Market Outlook**:
The market is expected to be choppy or flat tomorrow.
No clear trade setup recommended. Wait for volatility.

📰 **News Context**:
{reason}

📊 **Model Metrics**:
• Volatility (σ): {result.get('volatility', 0):.4f}
• Threshold: {result.get('threshold', 0):.4f}
• Model Confidence: {"High" if abs(result['pred_pct']) > result.get('threshold', 0)*200 else "Medium"}

📜 **Recent History**: {history_summary}

*Analysis based on live dynamic model training.*
*DYOR.*
        """
    
    # Log pick to database
    db.add_pick(
        ticker=result['ticker'],
        direction=result['direction'],
        pred_pct=result['pred_pct'],
        volatility=result.get('volatility', 0.0),
        current_price_usd=result['current_price_usd'],
        reason=reason
    )
    print(f"✅ Logged {result['ticker']} ({result['direction']}) to database")

    # Throttle before sending telegram
    print("Sleeping for 20s before sending Telegram...")
    time.sleep(20)

    # 4. Send to Telegram (Direct call - no Gemini overhead)
    print("Sending to Telegram...")
    telegram_status = send_telegram_message(final_message)
    print(f"Telegram Status: {telegram_status}")

if __name__ == "__main__":
    main()
