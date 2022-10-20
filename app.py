import flask
import pickle
from flask import Flask,request,app
from flask import Response
from app_logger import log
#from flask_cors import CORS
from flask import jsonify,url_for,render_template
import joblib
import numpy as np


app=Flask(__name__)
model=joblib.load(open('dust_prediction_rfr.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = [np.array(data)]
       
        output = model.predict(data)[0]
        log.info('Prediction done for Regression model')
        if output > 50:
            return render_template('index.html', prediction_text2="Value of Dust is {:.4f} ---- Warning!!! High hazard rating".format(output))
        else:
            return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(output))
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('index.html', prediction_text2="Check the Input again!!!")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    ''' for direct API calls through request'''
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output= model.predict(new_data)[0]
    return jsonify(output)
    

if __name__=='__main__':
    app.run(debug=True)

