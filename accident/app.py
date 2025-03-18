from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

filename='accident.pkl'
with open(filename,'rb') as f:
    model=pickle.load(f)
    
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No file part'})
    
    file=request.files['file']
    
    if file.filename=='':
        return jsonify({'error':'No selected file'})
    
    if file:
        df=pd.read_csv(file)
        
        sc=StandardScaler()
        df=sc.fit_transform(df)
        
        prediction = model.predict(df)
        prediction_text = 'Yes' if prediction[0] == 1 else 'No'
        
        return render_template('index.html', prediction_result=f"Survival Prediction: Yes/No: {prediction_text}")

    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."})

if __name__ == '__main__':
    app.run(debug=True)
