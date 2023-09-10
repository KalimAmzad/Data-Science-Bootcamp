import uvicorn
from fastapi import FastAPI
from PropertyVariables import ProperyPricePred
import pandas as pd
import joblib

# 1.  Creating the App object
PropertyPricePredApp = FastAPI()
# app = FastAPI()

# 2.  Load the model from disk
fileName = 'property_price_prediction_voting.sav'
loaded_model = joblib.load(fileName)



# 3. Index route, opens automatically on http://127.0.0.1:8000
@PropertyPricePredApp.get('/')
def index():
    return {'message': 'Hello, World!'}


@PropertyPricePredApp.get('/blog/{blog_id}')
def blog(blog_id: int):
    msg = f"you requested the blog Id: {blog_id}"
    return {"message": msg}


@PropertyPricePredApp.post('/add')
def addition(a: float, b: float):
    return a+b

@PropertyPricePredApp.post('/minus')
def subst(a: float, b: float):
    print(type(a), type(b))
    return abs(a - b)





# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted price with the confidence (http://127.0.0.1:8000/predict)
@PropertyPricePredApp.post('/predict')
def predict_price(data: ProperyPricePred):
    data = data.dict()
    print(data)
    data = pd.DataFrame([data])
    print(data.head())

    prediction = loaded_model.predict(data)
    print(str(prediction))
    return "The predicted price: " + str(prediction)


# # 5. Run the API with uvicorn
# #    Will run on http://127.0.0.1:8005
if __name__ == '__main__':
    uvicorn.run("application:PropertyPricePredApp",host='0.0.0.0', port=8005, reload=True, debug=True, workers=3)
