import flask
import pickle
from flask import Flask,request,app
from flask import Response
#from flask_cors import CORS
from flask import jsonify,url_for,render_template
import joblib
import numpy as np


app=Flask(__name__)
model=joblib.load(open('dust_prediction_rfr.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    ''' for direct API calls through request'''
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output= model.predict(final_features)[0]
    print(output)
    return render_template('home.html',prediction_text="Dust_predicted is {}".format(output))
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

