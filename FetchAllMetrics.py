import requests
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_url = os.getenv("WEARABLE_BASE_URL", "https://app-eks.gonoise.com")
ACCESS_TOKEN = os.getenv(
    "WEARABLE_ACCESS_TOKEN",
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo0NDI3MjQwOSwiZGV2aWNlX2lkIjo2NSwic3Vic2NyaXB0aW9uX3R5cGUiOiJiYXNpYyIsIndhdGNoX3N1Yl90eXBlIjoiZW50cnkiLCJpYXQiOjE3NzI5MjM3MzYsImV4cCI6MTc3MzUyODUzNn0.Xi5wPlyLc_nbe62p5QtkPbHodNpTXBO3PRtbVbOD2Tg"
)

endpoints = {
    "sleep": "/watch/sleep/v1/history",
    "spo2": "/watch/blood_oxygen/v1/history",
    "heart_rates": "/watch/heart_rates/v1/history"
}

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

params = {
    "history_type": "daily"
}

def run_fetch_cycle(single_run: bool = False):
    while True:
        for name, path in endpoints.items():
            url = f"{base_url}{path}"
            print(f"Fetching {name} data from {url}...")
            
            try:
                headers["epoch-time"] = str(int(time.time() * 1000))
                response = requests.get(url, headers=headers, params=params, timeout=15)
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    filename = os.path.join(BASE_DIR, f"{name}_response.json")
                    with open(filename, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved raw {name} data to {filename}\n")
                else:
                    print(f"Failed to fetch {name}. Response:\n{response.text[:200]}\n")
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}\n")

        if single_run or os.getenv("SINGLE_RUN", "0") == "1":
            print("Single run completed.")
            break

        print("Finished fetching all metrics. Waiting 60 seconds before next fetch...")
        time.sleep(60)

if __name__ == "__main__":
    run_fetch_cycle(single_run=True)
