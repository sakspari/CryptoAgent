from agno.agent import Agent
from agno.models.google import Gemini
import os
import requests

def send_telegram_message(message: str) -> str:
    """
    Sends a message to the configured Telegram chat.
    
    Args:
        message (str): The message to send.
        
    Returns:
        str: Status of the message sending.
    """
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return "Error: Telegram credentials not found in environment variables."
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return "Message sent successfully."
        else:
            return f"Failed to send message: {response.text}"
    except Exception as e:
        return f"Error sending message: {e}"

# Create the Telegram Agent
# Create the Telegram Agent
telegram_agent = Agent(
    name="Telegram Agent",
    role="Messenger",
    instructions="You are a messenger. Your only job is to send the provided message to Telegram using the tool.",
    tools=[send_telegram_message],
    model=Gemini(id=os.getenv("GEMINI_MODEL_ID", "gemini-flash-latest")),
    retries=int(os.getenv("AGENT_RETRIES", 3)),
    delay_between_retries=int(os.getenv("RETRY_DELAY", 5)),
    markdown=True,
)
