import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Concrete Compressive Strength Prediction using Machine Learning**!
""")
st.write('---')
image=Image.open(r'maxresdefault.jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"Concrete_Data.csv")
req_col_names = ["Cement (kg/m3)", "BlastFurnaceSlag (kg/m3)", "FlyAsh (kg/m3)", "Water (kg/m3)", "Superplasticizer (kg/m3)",
                 "CoarseAggregate (kg/m3)", "FineAggregare (kg/m3)", "Age (days)", "CC_Strength (kg/m3)"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('data information')
data.head()
data.isna().sum()
corr = data.corr()
st.dataframe(data)

X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]          # Target - Last Column
print(X)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    Cement = st.sidebar.slider('Cement (kg/m3)', 102,540,200)
    BlastFurnaceSlag = st.sidebar.slider('BlastFurnaceSlag (kg/m3)',0,359,300)
    FlyAsh = st.sidebar.slider('FlyAsh (kg/m3)', 0,200,150)
    Water = st.sidebar.slider('Water (kg/m3)', 121,247,200)
    Superplasticizer = st.sidebar.slider('Superplasticizer (kg/m3)', 0,32,15)
    CoarseAggregate = st.sidebar.slider('CoarseAggregate (kg/m3)', 801,1145,900)
    FineAggregare = st.sidebar.slider('FineAggregare (kg/m3)', 594,992,800)
    Age = st.sidebar.slider('Age (days)', 1,365,200)

    data_user = {'Cement': Cement,
            'BlastFurnaceSlag': BlastFurnaceSlag,
            'FlyAsh': FlyAsh,
            'Water': Water,
            'Superplasticizer': Superplasticizer,
            'CoarseAggregate': CoarseAggregate,
            'FineAggregare': FineAggregare,
            'Age': Age}
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')




# Reads in saved classification model
import pickle
load_clf = pickle.load(open('concrete_rfr.pkl', 'rb'))
st.header('Prediction of Concrete Compressive Strength (Mpa)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
