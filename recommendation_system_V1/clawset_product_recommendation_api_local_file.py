from fastapi import FastAPI, HTTPException, Form,UploadFile,File
from ultralytics import YOLOWorld
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import spacy
import cv2
import math
import requests



# Load SpaCy model and product descriptions
nlp = spacy.load("en_core_web_md")
df = pd.read_pickle("clawset_product_descriptions2.pkl")

# Preprocess and vectorize product descriptions
processed_descriptions = [nlp(desc.lower()) for desc in df['Description']]
similarity_matrix = cosine_similarity([doc.vector for doc in processed_descriptions])



# Load the pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8x-worldv2.pt")
classNames = ["hat", "collar belt", "leash", "harness", "collar", "bag", "shirt", "bed", "Scout", "scarf", "bed"]
model.set_classes(classNames)



app = FastAPI()

def get_recommendations(query, k):
    if query:
        query_doc = nlp(query.lower())
        query_vector = query_doc.vector
        sim_scores = cosine_similarity([query_vector], [doc.vector for doc in processed_descriptions])[0]
        top_indices = sim_scores.argsort()[-k:][::-1]  # Get top k similar descriptions
        recommendations = []
        counter = 1
        # print(top_indices)
        # print(df.iloc[top_indices]['SKU'])
        # recommendations = [
        #         {
        #             "product_name": df.iloc[top_indices]['Description'],
        #             "sku": df.iloc[top_indices]['SKU'],
        #             "category_name": df.iloc[top_indices]['Categories'],
        #             "confidence": float(sim_scores[i])
        #         }
        #     ]


        result_string = ','.join(df.iloc[top_indices]['SKU'].astype(str).tolist())

        print(result_string)

        url = f"https://clawset.co/en/wp-json/wc/v3/products?sku={result_string}"
        headers = {
            "Authorization": "Basic Y2tfMTQwODY1ODBkN2M4MjBhMjZlNmVkZTFkNzQ2MWRkMDE4OTUyNjljYjpjc183YmZhOTM0MzMyYWYxMDYyYWMwMDZlYjdhY2ZkZjE4YzMyNDNlY2Nl",
            "Cookie": "tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C3d764a25107975d6f2c5af1053f13e6e75bf624f56612e1550ac024e6a32dedd; wp-wpml_current_admin_language_d41d8cd98f00b204e9800998ecf8427e=th; wp-wpml_current_language=en; tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C8fdb2a74357f47a130f79d83fa1d71be851dc1029c8bef8393cee16a0fe4b999; tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C8fdb2a74357f47a130f79d83fa1d71be851dc1029c8bef8393cee16a0fe4b999"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and not data:
                print("The response is an empty list.")
                return []
            else:
                print('********* Data collected from api')
                return data
        else:
            print(f"Failed to retrieve data: {response.status_code}")

        # for i in top_indices:
        #     sku = df.iloc[i]['SKU']
            # url = f"https://clawset.co/en/wp-json/wc/v3/products?sku={sku}"
            # headers = {
            #     "Authorization": "Basic Y2tfMTQwODY1ODBkN2M4MjBhMjZlNmVkZTFkNzQ2MWRkMDE4OTUyNjljYjpjc183YmZhOTM0MzMyYWYxMDYyYWMwMDZlYjdhY2ZkZjE4YzMyNDNlY2Nl",
            #     "Cookie": "tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C3d764a25107975d6f2c5af1053f13e6e75bf624f56612e1550ac024e6a32dedd; wp-wpml_current_admin_language_d41d8cd98f00b204e9800998ecf8427e=th; wp-wpml_current_language=en; tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C8fdb2a74357f47a130f79d83fa1d71be851dc1029c8bef8393cee16a0fe4b999; tinvwl_wishlists_data_counter=0; wcml_client_country=PK; wfwaf-authcookie-b6260136d1434a758a13777e8974bb44=451%7Cadministrator%7Cmanage_options%2Cunfiltered_html%2Cedit_others_posts%2Cupload_files%2Cpublish_posts%2Cedit_posts%2Cread%7C8fdb2a74357f47a130f79d83fa1d71be851dc1029c8bef8393cee16a0fe4b999"
            # }
            # response = requests.get(url, headers=headers)
            # if response.status_code == 200:
            #     data = response.json()
            #     if isinstance(data, list) and not data:
            #         print("The response is an empty list.")
            #     else:
            #         if counter == 10:
            #             break
            #         else:
            #             print('********* Data collected from api')
            #             recommendations.append(data)
            #             counter+=1
            # else:
            #     print(f"Failed to retrieve data: {response.status_code}")
        #     # recommendations = [
        #     #     {
        #     #         "product_name": df.iloc[i]['Description'],
        #     #         "sku": df.iloc[i]['SKU'],
        #     #         "category_name": df.iloc[i]['Categories'],
        #     #         "confidence": float(sim_scores[i])
        #     #     }
        #     #     for i in top_indices
        #     # ]
        # return recommendations
    else:
        return []

@app.post("/image_based_recommandation_system")
async def image(image_file: UploadFile = File(...)):
    
    
    
    image_bytes = await image_file.read()
        
        # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode numpy array into image format (BGR)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
     
    # Perform prediction
    results = model(img, stream=True)
    predictions = []
    # Process results and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            print('confidence --- ', conf)
            
            cls = box.cls[0]
            name = classNames[int(cls)]
            # predictions.append({name: conf})
            predictions.append(name)

            print('name --- ', name)
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{name} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    predictions = list(set(predictions))
    print('prediction --- ', predictions)
    print("----------------")
    print(','.join(predictions))

    prediction_response=','.join(predictions)
    print('prediction_response: ', prediction_response)
    recommended_products=get_recommendations(prediction_response, k=15)
    # recommended_products['prediction_response'] = prediction_response
    print('prediction_response', prediction_response)

    # print('recommended_products', recommended_products)
    if recommended_products:
            return {'detected_product': prediction_response, "recommendations": recommended_products }
    else:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    

@app.post("/recommendations/")
# async def recommendations(query: str = Form(...), k: int = Form(5)):
async def recommendations(query: str = Form(...)):
    k  =15
    recommended_products = get_recommendations(query, k)
    if recommended_products:
        return {"recommendations": recommended_products}
    else:
        raise HTTPException(status_code=404, detail="No recommendations found.")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3535)