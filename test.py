
import requests

url = "https://global-power-plant-capacity-prediction.onrender.com/predict"

plant_data = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "country": "IND",
    "primary_fuel": "Solar",
    "commissioning_year": 2015,
    "other_fuel1": None,
    "other_fuel2": None,
    "other_fuel3": None,
    "estimated_generation_gwh": 120.5
}

response = requests.post(url, json=plant_data)

print("Status code:", response.status_code)
print("Response:", response.json())
