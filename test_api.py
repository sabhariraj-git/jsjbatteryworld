import requests

url = "http://127.0.0.1:5000/predict"
data = {"url": "youtube.com"}

try:
    response = requests.post(url, json=data, timeout=5)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
