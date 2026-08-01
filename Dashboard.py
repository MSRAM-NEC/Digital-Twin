import os
import json
import requests
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = os.getenv("WEARABLE_API_URL", "https://app-eks.gonoise.com/core/dashboard")
ACCESS_TOKEN = os.getenv(
    "WEARABLE_ACCESS_TOKEN",
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo0NDI3MjQwOSwiZGV2aWNlX2lkIjo2NSwic3Vic2NyaXB0aW9uX3R5cGUiOiJiYXNpYyIsIndhdGNoX3N1Yl90eXBlIjoiZW50cnkiLCJpYXQiOjE3NzI5MjM3MzYsImV4cCI6MTc3MzUyODUzNn0.Xi5wPlyLc_nbe62p5QtkPbHodNpTXBO3PRtbVbOD2Tg"
)

headers = {
    "content-type": "application/json",
    "version": "718",
    "version-name": "5.0.6",
    "user-agent": "Android 15(Redmi;25098RA98I;5.0.6)",
    "device-model": "25098RA98I",
    "device-manufacturer": "Redmi",
    "os-version": "15",
    "platform": "android",
    "country": "IN",
    "accept-language": "en",
    "epoch-time": str(int(time.time() * 1000)),
    "access-token": ACCESS_TOKEN,
    "device-id": "65",
    "device-type": "icon_buzz",
    "mac-id": "E3:78:0F:C6:5C:6F",
    "wearable-type": "watch",
    "timezone": "Asia/Kolkata",
    "offset": "330"
}

def fetch_dashboard():
    try:
        response = requests.get(API_URL, headers=headers, timeout=15)
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            data = response.json()
            out_file = os.path.join(BASE_DIR, "dashboard_response.json")
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved raw JSON to {out_file}")
            return data
        else:
            print(f"API request returned status code {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print("Failed to fetch dashboard metric:", e)

if __name__ == "__main__":
    fetch_dashboard()
