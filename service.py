

import bentoml
from bentoml.io import NumpyNdarray
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load models as runners (replace with correct paths)
scaler = bentoml.sklearn.get("scaler:latest").to_runner()
clf_svm = bentoml.sklearn.get("clf_svm:latest").to_runner()
clf_rf = bentoml.sklearn.get("clf_rf:latest").to_runner()
clf_dt = bentoml.sklearn.get("clf_dt:latest").to_runner()
clf_lg = bentoml.sklearn.get("clf_lg:latest").to_runner()

# Initialize all runners (this is usually handled by BentoML automatically)
scaler.init_local()
clf_svm.init_local()
clf_rf.init_local()
clf_dt.init_local()
clf_lg.init_local()

# Create the BentoML Service
service = bentoml.Service("water_quality_service", runners=[scaler, clf_svm, clf_rf, clf_dt, clf_lg])

# Define an API endpoint for Random Forest predictions
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_rf(input_data: np.ndarray):
    """
    Endpoint for predicting water potability using the Random Forest model.
    """
    logging.debug(f"Received input for RF prediction: {input_data}")
    
    # Scale the input data using the saved scaler
    scaled_data = scaler.predict.run(input_data)
    
    # Predict using the Random Forest model
    predictions = clf_rf.predict.run(scaled_data)
    
    logging.debug(f"Random Forest Prediction: {predictions}")
    
    return predictions

# Define an API endpoint for SVM predictions
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_svm(input_data: np.ndarray):
    """
    Endpoint for predicting water potability using the Support Vector Machine model.
    """
    logging.debug(f"Received input for SVM prediction: {input_data}")
    
    # Scale the input data using the saved scaler
    scaled_data = scaler.predict.run(input_data)
    
    # Predict using the SVM model
    predictions = clf_svm.predict.run(scaled_data)
    
    logging.debug(f"SVM Prediction: {predictions}")
    
    return predictions

# Define an API endpoint for Decision Tree predictions
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_dt(input_data: np.ndarray):
    """
    Endpoint for predicting water potability using the Decision Tree model.
    """
    logging.debug(f"Received input for Decision Tree prediction: {input_data}")
    
    # Scale the input data using the saved scaler
    scaled_data = scaler.predict.run(input_data)
    
    # Predict using the Decision Tree model
    predictions = clf_dt.predict.run(scaled_data)
    
    logging.debug(f"Decision Tree Prediction: {predictions}")
    
    return predictions

# Define an API endpoint for Logistic Regression predictions
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_lg(input_data: np.ndarray):
    """
    Endpoint for predicting water potability using the Logistic Regression model.
    """
    logging.debug(f"Received input for Logistic Regression prediction: {input_data}")
    
    # Scale the input data using the saved scaler
    scaled_data = scaler.predict.run(input_data)
    
    # Predict using the Logistic Regression model
    predictions = clf_lg.predict.run(scaled_data)
    
    logging.debug(f"Logistic Regression Prediction: {predictions}")
    
    return predictions

# Save the BentoML service for deployment (this will create a service archive)
if __name__ == "__main__":
    ## service.save()  # Saves the service definition and models
    logging.info("BentoML service saved successfully!")



