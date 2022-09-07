import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Payload':1500, 'Type':0, 'Dist_To_Travel':1,'region/non_region':1})

print(r.json())