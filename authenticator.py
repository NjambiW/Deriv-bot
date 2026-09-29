import requests
from config import *

def get_websocket_url():
    """connects to the deriv api and authenticates it"""
    headers = {
        "Deriv-App-ID": APP_ID,
        "Authorization": f"Bearer {PATAPI}"
    }

    websocketUrl = f"https://api.derivws.com/trading/v1/options/accounts/{CLIENTid}/otp"

    response = requests.post(
        url=websocketUrl,
        headers=headers
    )

    response.raise_for_status()

    data = response.json()

    ws_url = data["data"]["url"]

    print(ws_url)
    print(response.status_code)
    print(response.text)
    return data["data"]["url"]