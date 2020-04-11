import requests

url = 'http://127.0.0.1:8000/api/home'

r = requests.post(url, json={'day': 3})
print(int(float(r.json().get('prediction'))))
print(r.status_code)