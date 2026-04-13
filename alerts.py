import requests
from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
WHATSAPP_FROM = 'whatsapp:+14155238886'
WHATSAPP_TO = f"whatsapp:{os.getenv('TWILIO_TO_NUMBER')}"
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN')
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY')

def send_whatsapp(message: str):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=WHATSAPP_FROM,
            to=WHATSAPP_TO
        )
        # Poll for final status (sandbox messages often go undelivered silently)
        import time
        time.sleep(3)
        updated = client.messages(msg.sid).fetch()
        status  = updated.status   # 'delivered', 'undelivered', 'failed', 'sent', 'queued'
        if status in ('delivered', 'sent', 'queued', 'read'):
            print(f'✅ WhatsApp sent: {msg.sid} [{status}]')
            return True
        else:
            print(
                f'❌ WhatsApp undelivered (status={status}). '
                f'Twilio sandbox may have expired — '
                f'send "join <keyword>" to +14155238886 on WhatsApp to re-activate.'
            )
            return False
    except Exception as e:
        print(f'❌ WhatsApp error: {e}')
        return False

def send_push(title: str, message: str):
    try:
        r = requests.post('https://api.pushover.net/1/messages.json', data={
            'token': PUSHOVER_APP_TOKEN,
            'user': PUSHOVER_USER_KEY,
            'title': title,
            'message': message,
            'sound': 'cashregister'
        })
        if r.status_code == 200:
            print(f'✅ Push sent')
            return True
        else:
            print(f'❌ Push failed ({r.status_code}): {r.text}')
            return False
    except Exception as e:
        print(f'❌ Push error: {e}')
        return False

def send_alert(ticker: str, signal: str, price: float,
               entry_low: float, entry_high: float,
               targets: list, stop: float,
               reason: str, confidence: int) -> bool:
    """
    Send a trade alert via WhatsApp (always) and Pushover (if configured).
    targets: list of [T1, T2, T3] floats from the analyzer.
    Only targets above the entry price are shown (filters out bad T1 levels).
    Returns True if WhatsApp delivery succeeded, False otherwise.
    """
    emoji = 'BUY' if signal == 'BUY' else 'SELL' if signal == 'SELL' else 'HOLD'
    entry_mid = (entry_low + entry_high) / 2

    # Keep only targets that are actually above entry (profit targets, not resistance below)
    valid_targets = [t for t in (targets if isinstance(targets, list) else [targets])
                     if t > entry_mid]

    # Build target lines: T1 +3.5%, T2 +9.2%, T3 +18.4%
    target_lines = []
    labels = ['T1', 'T2', 'T3']
    for i, t in enumerate(valid_targets[:3]):
        pct = (t - entry_mid) / entry_mid * 100 if entry_mid > 0 else 0
        target_lines.append(f'  {labels[i]}: ${t:.2f} (+{pct:.1f}%)')

    if not target_lines:
        target_lines = ['  No target above entry — skip this trade']

    stop_pct = (entry_mid - stop) / entry_mid * 100 if entry_mid > 0 else 0
    targets_str = '\n'.join(target_lines)

    message = (
        f'Stock AI Agent - {emoji}\n'
        f'Ticker:     {ticker} at ${price:.2f}\n'
        f'Entry zone: ${entry_low:.2f} – ${entry_high:.2f}\n'
        f'Targets:\n{targets_str}\n'
        f'Stop loss:  ${stop:.2f} (-{stop_pct:.1f}%)\n'
        f'Confidence: {confidence}%\n'
        f'Reason: {reason}'
    )

    whatsapp_ok = send_whatsapp(message)

    if PUSHOVER_APP_TOKEN and PUSHOVER_APP_TOKEN != 'your_key_here':
        send_push(f'Stock AI Agent - {signal} {ticker}', message)

    return bool(whatsapp_ok)

if __name__ == '__main__':
    send_alert(
        ticker='BZAI',
        signal='BUY',
        price=1.79,
        entry_low=1.75,
        entry_high=1.82,
        targets=[1.95, 2.10, 2.35],
        stop=1.65,
        reason='RSI oversold + volume spike 3.7x',
        confidence=72
    )
