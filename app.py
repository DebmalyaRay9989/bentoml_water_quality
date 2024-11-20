
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import bentoml
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create FastAPI app
app = FastAPI()

# Define the input structure (using Pydantic)
class WaterQualityInput(BaseModel):
    ph: float
    Sulfate: float
    Trihalomethanes: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float

# Load the models from BentoML
scaler_runner = bentoml.sklearn.get("scaler:latest").to_runner()
clf_rf_runner = bentoml.sklearn.get("clf_rf:latest").to_runner()
clf_svm_runner = bentoml.sklearn.get("clf_svm:latest").to_runner()
clf_dt_runner = bentoml.sklearn.get("clf_dt:latest").to_runner()
clf_lg_runner = bentoml.sklearn.get("clf_lg:latest").to_runner()

# Initialize the models locally
scaler_runner.init_local()
clf_rf_runner.init_local()
clf_svm_runner.init_local()
clf_dt_runner.init_local()
clf_lg_runner.init_local()

@app.post("/predict_rf/")
async def predict_rf(input_data: WaterQualityInput):
    input_features = np.array([[input_data.ph, input_data.Sulfate, input_data.Trihalomethanes,
                                input_data.feature4, input_data.feature5, input_data.feature6, 
                                input_data.feature7, input_data.feature8, input_data.feature9]])
    
    try:
        # Apply scaling
        scaled_features = scaler_runner.predict.run(input_features)
        
        # Make prediction with RandomForest
        prediction = clf_rf_runner.predict.run(scaled_features)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_svm/")
async def predict_svm(input_data: WaterQualityInput):
    input_features = np.array([[input_data.ph, input_data.Sulfate, input_data.Trihalomethanes,
                                input_data.feature4, input_data.feature5, input_data.feature6, 
                                input_data.feature7, input_data.feature8, input_data.feature9]])
    
    try:
        # Apply scaling
        scaled_features = scaler_runner.predict.run(input_features)
        
        # Make prediction with SVM
        prediction = clf_svm_runner.predict.run(scaled_features)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_dt/")
async def predict_dt(input_data: WaterQualityInput):
    input_features = np.array([[input_data.ph, input_data.Sulfate, input_data.Trihalomethanes,
                                input_data.feature4, input_data.feature5, input_data.feature6, 
                                input_data.feature7, input_data.feature8, input_data.feature9]])
    
    try:
        # Apply scaling
        scaled_features = scaler_runner.predict.run(input_features)
        
        # Make prediction with Decision Tree
        prediction = clf_dt_runner.predict.run(scaled_features)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_lg/")
async def predict_lg(input_data: WaterQualityInput):
    input_features = np.array([[input_data.ph, input_data.Sulfate, input_data.Trihalomethanes,
                                input_data.feature4, input_data.feature5, input_data.feature6, 
                                input_data.feature7, input_data.feature8, input_data.feature9]])
    
    try:
        # Apply scaling
        scaled_features = scaler_runner.predict.run(input_features)
        
        # Make prediction with Logistic Regression
        prediction = clf_lg_runner.predict.run(scaled_features)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Water Quality Prediction API is running!"}



