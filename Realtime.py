import requests
import json
import pandas as pd

# API endpoint discovered from mitmproxy
url = "https://app-eks.gonoise.com/watch/heart_rates/v1/history"

# Query parameters
params = {
    "history_type": "daily"
}

# Headers captured from the request
headers = {
    "access-token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo0NDI3MjQwOSwiZGV2aWNlX2lkIjo2NSwic3Vic2NyaXB0aW9uX3R5cGUiOiJiYXNpYyIsIndhdGNoX3N1Yl90eXBlIjoiZW50cnkiLCJpYXQiOjE3NzI5MjM3MzYsImV4cCI6MTc3MzUyODUzNn0.Xi5wPlyLc_nbe62p5QtkPbHodNpTXBO3PRtbVbOD2Tg",
    "platform": "android",
    "device-type": "icon_buzz",
    "accept": "application/json"
}

# Send request
response = requests.get(url, headers=headers, params=params)

print("Status Code:", response.status_code)

# Convert response to JSON
data = response.json()

print("\nAPI Response:")
print(json.dumps(data, indent=2))

# Save data to CSV if present
if data.get("success") and "data" in data and "heart_rates" in data["data"]:
    heart_rates = data["data"]["heart_rates"]
    
    # Store the daily level data (excluding the nested hourly array)
    daily_records = []
    hourly_records = []
    
    for day in heart_rates:
        daily_record = {k: v for k, v in day.items() if k != "hourly_break_up"}
        daily_records.append(daily_record)
        
        if "hourly_break_up" in day:
            hourly_records.extend(day["hourly_break_up"])

    # Save daily data
    if daily_records:
        df_daily = pd.DataFrame(daily_records)
        df_daily.to_csv("heart_rate_history_daily.csv", index=False)
        print("\nSaved daily summary to heart_rate_history_daily.csv")
        
    # Save hourly data
    if hourly_records:
        df_hourly = pd.DataFrame(hourly_records)
        df_hourly.to_csv("heart_rate_history_hourly.csv", index=False)
        print("Saved hourly break-up to heart_rate_history_hourly.csv")
else:
    print("\nNo heart rate data found.")