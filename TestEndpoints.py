import requests
import time

base_url = "https://app-eks.gonoise.com"

endpoints = [
    "/watch/sport/v1/history",
    "/watch/sports/v1/history",
    "/watch/workout/v1/history",
    "/watch/workouts/v1/history",
    "/watch/movement/v1/history",
    "/watch/pedometer_steps/v1/history",
    "/watch/fitness/v1/history"
]

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
    "access-token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo0NDI3MjQwOSwiZGV2aWNlX2lkIjo2NSwic3Vic2NyaXB0aW9uX3R5cGUiOiJiYXNpYyIsIndhdGNoX3N1Yl90eXBlIjoiZW50cnkiLCJpYXQiOjE3NzI5MjM3MzYsImV4cCI6MTc3MzUyODUzNn0.Xi5wPlyLc_nbe62p5QtkPbHodNpTXBO3PRtbVbOD2Tg",
    "device-id": "65",
    "device-type": "icon_buzz",
    "mac-id": "E3:78:0F:C6:5C:6F",
    "wearable-type": "watch",
    "timezone": "Asia/Kolkata",
    "offset": "330"
}
params = {"history_type": "daily"}

for path in endpoints:
    url = f"{base_url}{path}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print(f"SUCCESS: {path}")
        with open(f"response_{path.split('/')[2]}.json", "w") as f:
            f.write(response.text)
    elif response.status_code != 404:
        print(f"OTHER: {path} -> {response.status_code}")
