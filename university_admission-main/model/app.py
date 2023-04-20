from flask import Flask,render_template,request, redirect,url_for
import pickle
import numpy as np
from scipy.special import expit

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')



@app.route('/result',methods=['POST','GET'])
def result():
    check = False
    if request.method=='POST':
        check=True
        int_features = [float(x) for x in request.form.values()]
        
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = expit(prediction[0])*100
        
        return render_template("index.html",check=check,prediction_text=output)
        
    elif request.method == 'GET':
        return redirect("/")


if __name__ == '__main__':
    app.run(debug=True)