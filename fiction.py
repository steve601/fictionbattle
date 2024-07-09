from flask import Flask,request,render_template
import pandas as pd   
import pickle

app = Flask(__name__)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

pre_path = 'elements\preprocessor.pkl'
model_path = 'elements\model.pkl'
# loading the objects

preprocessor = load_object(pre_path)
model = load_object(model_path)

@app.route('/')
def homepage():
    return render_template('fiction.html')

@app.route('/predict',methods=['POST'])
def do_pred(): 
    inp = [x for x in request.form.values()]
    columns = ['character', 'universe', 'strength', 'speed', 'intelligence',
       'specialabilities', 'weaknesses']
    inp_df = pd.DataFrame([inp],columns=columns)

    
    inp_df = preprocessor.transform(inp_df)
    
    pred = model.predict(inp_df)
    
    msg = 'Character 1 wins!' if pred == 1 else 'Character 2 wins!'
    
    return render_template('fiction.html',text=msg)

if __name__ == "__main__":
    app.run(debug=True)