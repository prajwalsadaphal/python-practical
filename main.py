import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler

#################################################################    


# Hospital_region_code_label = {'X':0, 'Y':1, 'Z':2} 


# Get the Keys
def get_value(val,my_dict):
    for key ,value in my_dict.items():
        if val == key:
            return value

# Find the Key From Dictionary
def get_key(val,my_dict):
    for key ,value in my_dict.items():
        if val == value:
            return key

def FeatureScaler(df):
    min_max = MinMaxScaler()
    df = pd.DataFrame(min_max.fit_transform(df), columns=df.columns)
    return df


def main():
    st.title('HealthCare Prediction App')

    #Menu
    menu = ['Prediction', 'About']
    choice = st.sidebar.selectbox('Select Activities', menu)

    if choice == 'Prediction':
        City_Code_Hospital = st.slider('What is the City Code of Hospital?', 1, 13)
        Hospital_region_code = st.slider("What is the Hospital region code",0,3)
        Available_Extra_Rooms_in_Hospital = st.slider("How Much Extra Rooms are Available in Hospital ?", 0, 24)
        Department	    = st.slider("Which department do you Want?", 0, 4)
        Ward_Type     = st.slider('Which Ward do you want (Ward Code)?', 0, 5)
        Ward_Facility_Code  = st.slider("Which Ward Facility do you want (Ward Facility Code)",0, 5)
        Bed_Grade  = st.slider('Which Bed Grade you want?',1, 4)
        City_Code_Patient = st.slider('What is the City Code of Patient?',1, 38)
        Type_of_Admission  = st.slider('What type of Admission do we Register?', 0, 2)
        Severity_of_Illness = st.slider('What is the Severity of Illness?', 0, 2)
        Visitors_with_Patient = st.slider("How many visitors are with the patient",1, 25)
        Age = st.slider("What Age range is the patient?", 0, 9)
        Admission_Deposit = st.slider("How much Admission Deposit did you pay?", 1800, 11008)

    #Data That Will Use For Prediction
    input_data = [City_Code_Hospital, Hospital_region_code, Available_Extra_Rooms_in_Hospital, Department, Ward_Type, Ward_Facility_Code, Bed_Grade, City_Code_Patient, Type_of_Admission, Severity_of_Illness, Visitors_with_Patient, Age, Admission_Deposit]

    # Converted the input list to a DataFrame
    df_input_data = pd.DataFrame(input_data)

    # Feature Scaled the O/P Dataframe for prediction
    output = FeatureScaler(df_input_data)

    input_for_model = np.array(output).reshape(1, -1)
    


     #Prediction
    if st.button("Predict!"):
        predictor = pickle.load(open("Random_Forest_Model.pkl", 'rb'))
        prediction = predictor.predict(input_for_model)
        # predict_proba = predictor.predict_proba(input_data)[:,1]
        # hasil = (str((np.around(float(predict_proba),3) * 100)) + '%')
        st.subheader('The Probability This Loan Will Be Default is: ')        
        # st.subheader('The Probability This Loan Will Be Default is: ' + prediction)










if __name__ == '__main__':
    main()