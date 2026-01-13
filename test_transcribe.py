import requests

url = "http://localhost:8000/transcribe"
files = {"file": open("sermon.ogg", "rb")}
data = {"include_timestamps": False}

response = requests.post(url, files=files, data=data)
print("Response status:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Transcription:", result.get("text", "No text found"))
else:
    print("Error:", response.text)