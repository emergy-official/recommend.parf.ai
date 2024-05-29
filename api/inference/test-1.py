import requests  
import base64  
import json  
from PIL import Image, ImageDraw  
import matplotlib.pyplot as plt  
import numpy as np  
from io import BytesIO  

# The local path to your file

# The URL of your Flask API endpoint  
# url = 'http://localhost:8887/invocations'  
# url = 'http://127.0.0.1:9000/invocations'  
url = 'http://127.0.0.1:8080/invocations'  


data = json.dumps({"userId": 15587})  
# Set the appropriate headers for a JSON payload  
headers = {'Content-Type': 'application/json'}  

# Make the POST request  
response = requests.post(url, data=data, headers=headers)  

# If the request is successful, print the response  
if response.status_code == 200:  
    
    resp = response.json()
    print("Success:")  
    print(resp)
else:  
    print("Error:", response.status_code)  
    print(response.text)
    