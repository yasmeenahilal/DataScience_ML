# importing the libraries
from flask import Flask,render_template,request
import pickle

#Global variables
app=Flask(__name__)
loaded_model=pickle.load(open('Knn_model.pkl','rb'))

#user defined routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/prediction",methods=['POST'])
def predict():
    ph=request.form['ph']
    Temperature=request.form['Temperature']
    Taste=request.form['Taste']
    odor=request.form['Odor']
    Fat=request.form['Fat']
    Turbidity=request.form['Turbidity']
    colour=request.form['Colour']
    
    
    prediction=loaded_model.predict([[ph,Temperature,Taste,odor,
                                      Fat,Turbidity,colour]])[0]
    
    if prediction==0:
        prediction="Bad"
    elif prediction==1:
        prediction='Moderate'    
    else:
        prediction='Good'
        
    return render_template("index.html",output_prediction=prediction)


if __name__ =='__main__':
    app.run(debug=True)       


