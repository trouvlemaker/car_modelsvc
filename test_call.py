import cv2
import requests
import numpy as np

image = "/app/twincar-modelsvc-gateway/test_image/SA2019111882700_2019111815475000.png"

image_data = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

api = "http://localhost:8091/api/predict/valid/"

header = {
    'accept' : 'application/json',
    'Content-Type' : 'application/json'
    }

data = {"image_data":image_data.tolist()}


response = requests.post(api, headers=header, json = data)

# convert output to numpy array

result = response.json()
# print(result)

# input_image = np.array(image_data)

# for key, value in result.items():
    
#     for box, polygon, label, score in value:

#         confidence = (np.round(score,2)*100)
#         label = f"{label} {confidence: .1f}%"        
#         input_image = draw_bbox(input_image, polygon, label = label, color=(0,255,0), thickness=2)    
#     input_image = cv2.cvtColor(input_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
#     cv2.imwrite("/app/twinreader-tcd-client/test_image/SA2019111882700_2019111815475000_res.png", input_image)