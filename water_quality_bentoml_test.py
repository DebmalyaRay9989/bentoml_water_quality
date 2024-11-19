
import bentoml
import pandas as pd
import numpy as np
import bentoml
from bentoml.io import PandasDataFrame, NumpyNdarray
import logging
import warnings

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Set up logging to suppress specific warning
logging.getLogger("bentoml._internal.runner.runner").setLevel(logging.ERROR)

# Issue a warning for using deprecated init_local() calls
warnings.warn("`init_local()` is for debugging and testing only. Remove it before deploying to production.", DeprecationWarning)


# Load models as runners
scaler = bentoml.sklearn.get("scaler:latest").to_runner()
clf_svm = bentoml.sklearn.get("clf_svm:latest").to_runner()
clf_rf = bentoml.sklearn.get("clf_rf:latest").to_runner()
clf_dt = bentoml.sklearn.get("clf_dt:latest").to_runner()
clf_lg = bentoml.sklearn.get("clf_lg:latest").to_runner()

scaler.init_local()
clf_svm.init_local()
clf_rf.init_local()
clf_dt.init_local()
clf_lg.init_local()

# Example warning if a model is missing (you can modify this check as needed)
if not scaler:
    warnings.warn("Scaler model not found. Make sure the model is saved and available.", UserWarning)
if not clf_rf:
    warnings.warn("Random Forest model not found. Make sure the model is saved and available.", UserWarning)


try:
    predictions = clf_rf.predict.run([[129.422921,18630.0,6.35246,592.885359,15.180013,4.500656,3.716080,298.082462,56.329076]])
    logging.debug(f"Predictions: {predictions}")

except Exception as e:
    logging.error(f"Error during prediction: {e}")

