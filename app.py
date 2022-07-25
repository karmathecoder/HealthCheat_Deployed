import numpy as np
from flask import Flask,request,jsonify, render_template
import pickle

app=Flask(__name__)
heart_svc = pickle.load(open('heart_svc/heart_svc.pkl','rb'))
diabetes_rfc = pickle.load(open('diabetes_rfc/diabetes_rfc.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart_page')
def heart():
    return render_template('heart_page.html')

@app.route('/heart_page_predict', methods=['POST'])
def predict_heart():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = heart_svc.predict(final_features)
    if prediction == [1]:
      return render_template('heart_page.html', prediction_text='High Risk Of Heart Disease')
    else:
      return render_template('heart_page.html', prediction_text='Low Risk Of Heart Disease')

@app.route('/diabaties_page')
def diabaties():
    return render_template('diabetes_page.html')

@app.route('/diabaties_page_predict',methods=['POST'])
def predict_diabetes():
  int_features = [float(x) for x in request.form.values()]
  final_features = [np.array(int_features)]
  prediction = diabetes_rfc.predict(final_features)
  if prediction == [1]:
    return render_template('diabetes_page.html', prediction_text='High Risk Of Diabetes')
  else:
    return render_template('diabetes_page.html', prediction_text='Low Risk Of Diabetes')

if __name__ == '__main__':
    app.run(debug=True)
