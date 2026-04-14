import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "")

# Pushover
PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# App
TICKER = os.getenv("TICKER", "BZAI")
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "60"))

# Circuit breaker
VIX_THRESHOLD      = float(os.getenv("VIX_THRESHOLD",      "25"))
SPY_DROP_THRESHOLD = float(os.getenv("SPY_DROP_THRESHOLD", "-1.5"))

# LangSmith — tracing is activated automatically when these are set in the environment.
# No explicit SDK calls needed; LangGraph picks them up on import.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "stock-ai-agent")
