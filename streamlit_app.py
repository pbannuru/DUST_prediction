import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('dust_prediction_rfr.pkl', 'rb')
regressor = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(MILL_IN_SERVICE,TOTAL_AIR_FLOW,COAL_FLOW,SO2,NOX,CHIMNEY_O2,BURNER_TILT,FLUE_GAS_TEMP):  
   
    prediction = regressor.predict(
        [[MILL_IN_SERVICE,TOTAL_AIR_FLOW,COAL_FLOW,SO2,NOX,CHIMNEY_O2,BURNER_TILT,FLUE_GAS_TEMP]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Dust_Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    #							   
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    MILL_IN_SERVICE = st.text_input("MILL IN SERVICE", "Type Here")
    TOTAL_AIR_FLOW = st.text_input("TOTAL AIR FLOW (TPH)", "Type Here")
    COAL_FLOW = st.text_input("COAL_FLOW", "Type Here")
    SO2 = st.text_input("SO2", "Type Here")
    NOX = st.text_input("NOX", "Type Here")
    CHIMNEY_O2 = st.text_input("CHIMNEY_O2", "Type Here")
    BURNER_TILT = st.text_input("BURNER_TILT", "Type Here")
    FLUE_GAS_TEMP = st.text_input("FLUE_GAS_TEMP", "Type Here")
    
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(MILL_IN_SERVICE,TOTAL_AIR_FLOW,COAL_FLOW,SO2,NOX,CHIMNEY_O2,BURNER_TILT,FLUE_GAS_TEMP)
    st.success(result)
     
if __name__=='__main__':
    main()