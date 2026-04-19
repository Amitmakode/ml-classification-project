# This file defines FastAPI app with prediction and health check endpoints

from fastapi import FastAPI, HTTPException  # fastapi framework
from pydantic import BaseModel  # request body validation
import sys  # system path manipulation
import os  # access env variables
from dotenv import load_dotenv  # load env variables

load_dotenv()  # load .env file

# add ml directory to path so predict imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))

from predict import predict  # import predict function

app = FastAPI(
    title="Online Shoppers Intention API",  # api title
    description="Predicts if a shopper will purchase or not",  # api description
    version="1.0.0"  # api version
)

# define input schema using pydantic — all 17 features
class ShopperInput(BaseModel):
    Administrative: int              # number of administrative pages visited
    Administrative_Duration: float   # time spent on administrative pages
    Informational: int               # number of informational pages visited
    Informational_Duration: float    # time spent on informational pages
    ProductRelated: int              # number of product related pages visited
    ProductRelated_Duration: float   # time spent on product related pages
    BounceRates: float               # bounce rate of visited pages
    ExitRates: float                 # exit rate of visited pages
    PageValues: float                # page value of visited pages
    SpecialDay: float                # closeness to special day
    Month: int                       # month of visit encoded as int
    OperatingSystems: int            # operating system of visitor
    Browser: int                     # browser of visitor
    Region: int                      # region of visitor
    TrafficType: int                 # traffic type of visitor
    VisitorType: int                 # visitor type encoded as int
    Weekend: int                     # 1 if weekend else 0

# define output schema
class PredictionOutput(BaseModel):
    prediction: int      # 0 or 1
    probability: float   # confidence score
    message: str         # human readable result

@app.get("/")  # root endpoint
def root():
    return {"status": "Online Shoppers Intention API is running"}  # health check

@app.get("/health")  # health check endpoint
def health():
    return {"status": "healthy"}  # return healthy status

@app.post("/predict", response_model=PredictionOutput)  # prediction endpoint
def predict_endpoint(data: ShopperInput):
    try:
        input_dict = data.model_dump()  # convert pydantic model to dict
        result = predict(input_dict)    # run prediction
        return result                   # return prediction result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # return error if prediction fails