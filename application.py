from flask import Flask,request,app,render_template
from flask import Response
# render template--finding url of html file
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# interact with pickel files

ridge_file=pickle.load(open('/config/workspace/Models/PredictionModel.pkl','rb'))
scaler_file=pickle.load(open('/config/workspace/Models/standardScaler.pkl','rb'))
# HomePage
# route for home page
@app.route("/")
def hello_world():
    return render_template('/config/workspace/templates/index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    result=""
    if request.method=="POST":

        Preg=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BP=float(request.form.get('BloodPressure'))
        St=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

    # now as new data is coming we will use pickle file on them
    # on testing data always only transform

        new_scale_data=scaler_file.transform([[Preg,Glucose,BP,St,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict_data=ridge_file.predict(new_scale_data)

        if predict_data[0]==1:
            result="Diabetic"
        else:
            result="Non-Diabetic"
        return render_template('single_prediction.html',result=result)
    else:
         return render_template('home.html')

    



if __name__=="__main__":
    app.run(host="0.0.0.0")
