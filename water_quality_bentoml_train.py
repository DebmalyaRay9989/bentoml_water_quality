from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import bentoml
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON
import numpy as np
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np

water_portability = pd.read_csv("water_potability.csv")
water_portability.info()
water_portability.isnull().sum()

def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample

median=water_portability.ph.median()
impute_nan(water_portability,"ph",median)
water_portability = water_portability.drop(columns=["ph","ph_median"])
water_portability = water_portability.rename(columns={"ph_random": "ph"})

median=water_portability.Sulfate.median()
impute_nan(water_portability,"Sulfate",median)
water_portability = water_portability.drop(columns=["Sulfate","Sulfate_median"])
water_portability = water_portability.rename(columns={"Sulfate_random": "Sulfate"})
median=water_portability.Trihalomethanes.median()
impute_nan(water_portability,"Trihalomethanes",median)
water_portability = water_portability.drop(columns=["Trihalomethanes","Trihalomethanes_median"])
water_portability = water_portability.rename(columns={"Trihalomethanes_random": "Trihalomethanes"})


X = water_portability.drop(columns = "Potability")
Y = water_portability["Potability"]

scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

bento_model_scaler = bentoml.sklearn.save_model("scaler", scaler)
print(f"Model saved: {bento_model_scaler}")


clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X, Y)
bento_model_svm = bentoml.sklearn.save_model("clf_svm", clf_svm)
print(f"Model saved: {bento_model_svm}")


clf_rf = RandomForestClassifier()
clf_rf.fit(X, Y)
bento_model_rf = bentoml.sklearn.save_model("clf_rf", clf_rf)
print(f"Model saved: {bento_model_rf}")

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X, Y)
bento_model_dt = bentoml.sklearn.save_model("clf_dt", clf_dt)
print(f"Model saved: {bento_model_dt}")

clf_lg = LogisticRegression()
clf_lg.fit(X, Y)
bento_model_lg = bentoml.sklearn.save_model("clf_lg", clf_lg)
print(f"Model saved: {bento_model_lg}")


