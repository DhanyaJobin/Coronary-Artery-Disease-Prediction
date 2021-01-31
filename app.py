from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict(): 
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    out = round(prediction[0])
    if out == 1: 
            output ='High'
    else: 
            output ='Low' 
    
    return render_template('home.html', prediction_text='Probability of Heart Disease is  $ {}'.format(output))
    
    
    
if __name__ == "__main__":
       app.run()