import tensorflow as tf
from fastapi import Request
import cv2
import pandas as pd
import numpy as np
import os

#loading model------------------------------
cnn_model = tf.keras.models.load_model('intel_model.h5')
rnn_model =tf.keras.models.load_model('intel_nlp.h5')
#-tokenisation----------------------------------------------
path = os.path.join('.', 'data', 'Training.csv')
data=pd.read_csv(path)

keywords=data.columns[:-2]
keywords=list(keywords)

classes=data.iloc[:,-2].unique()

vocab_map={}
for i in range(0,len(keywords)):
    vocab_map[i]=keywords[i].split("_")

def tokenize(x: str, vocab_hashmap: dict):
    X = np.zeros(len(vocab_hashmap))  
    if '_' in x:
        words = x.replace('_', ' ').split()
    else:
        words = x.split()
    
    for word in words:
        related_keys = set()
        
        for key, word_list in vocab_hashmap.items():
            if word in word_list:
                related_keys.add(key)
        for key in related_keys:
            X[key] = 1
    
    return X
#-------------------------------------------------
print("--")
#api


from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        print(file.filename)
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = cnn_model.predict(image)
        print("--------------------------",predictions)
        predicted_probability = float(predictions[0][0])

        if predicted_probability >= 0.5:
            predicted_class = "pneumonia"
        else:
            predicted_class = "normal"

        return {"class_name": predicted_class, "probability": predicted_probability}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_symptoms/")
async def predict_symptoms(request : Request):
    try:
        body = await request.json();
        tokenized_input = tokenize(body, vocab_map)
        X_input = np.array([tokenized_input])
        predictions = rnn_model.predict(X_input)
       
        top_three_indices = np.argsort(predictions[0])[-3:][::-1]
        top_three_diseases = [classes[i] for i in top_three_indices]
        
        return {"top_three_diseases": top_three_diseases}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
#uvicorn main:app --reload

