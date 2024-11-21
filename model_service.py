
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Load model runners for different classifiers
#scaler_runner = bentoml.sklearn.get("scaler:latest").to_runner()
clf_rf_runner = bentoml.sklearn.get("clf_rf:latest").to_runner()
clf_svm_runner = bentoml.sklearn.get("clf_svm:latest").to_runner()
clf_dt_runner = bentoml.sklearn.get("clf_dt:latest").to_runner()
clf_lg_runner = bentoml.sklearn.get("clf_lg:latest").to_runner()

# Create BentoML service and add the runners

service = bentoml.Service(
    "model_service", runners=[clf_rf_runner, clf_svm_runner, clf_dt_runner, clf_lg_runner]
)

#service = bentoml.Service(
#    "model_service", 
#    runners=[scaler_runner, clf_rf_runner, clf_svm_runner, clf_dt_runner, clf_lg_runner]
#)


# Define the API function for prediction
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    """
    This API function predicts using a specific model, say Random Forest in this case.
    You can modify this to dynamically choose the classifier based on input or other logic.
    """
    # First apply the scaler to the input data
    #scaled_input = scaler_runner.run(input_series)
    
    # Now make a prediction using one of the models, e.g., Random Forest
    result = clf_rf_runner.run(input_series)  # This is just an example, you can change it to another classifier

    return result



