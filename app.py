from flask import Flask, request
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df = df.drop(columns=['target'])

app = Flask(__name__)

# API end point@app.route('/', methods=['GET'])
@app.route('/')
def home():
    return "Please use one of the links: <br> \
        <a href='/setosa'>setosa</a>, <br> \
        <a href='/versicolor'>versicolor</a>, <br> \
        If interested, provide a (sepal length in cm) and b (petal length in cm) values in the link by adding '?a=1&b=1' <br> \
        X=1 and y=1 are set as default values unless specified otherwise."

@app.route('/<string:species>', methods=['GET'])
def perceptron(species):
    if species.lower() not in ['setosa', 'versicolor'] or species is None:
        return "Please use one of the links: <br> \
        <a href='/setosa'>setosa</a>, <br> \
        <a href='/versicolor'>versicolor</a>, <br> \
        If interested, provide a (sepal length in cm) and b (petal length in cm) values in the link by adding '?a=1&b=1' <br> \
        X=1 and y=1 are set as default values unless specified otherwise."
    
    a = request.args.get('a')
    b = request.args.get('b')
    
    try:
        a = float(a)
    except:
        a = 1
    if a <= 0:
        a = 1
        
    try:
        b = float(b)
    except:
        b = 1
    if b <= 0:
        b = 1
        
    X = df.iloc[:100, [0,2]].values
    y = df.iloc[:100, 4].values
    y = np.where(y == species.lower(), -1,1)
    
    per_clf = Perceptron()
    per_clf.fit(X,y)
    y_pred = per_clf.predict([[a, b]])
    result = True if y_pred[0] == -1 else False
    
    if result:
        return f"Given sepal length (a) {a} cm and petal length (b) {b} cm, it MIGHT BE {species}. <br> <a href='/'>HOME</a>"
    return f"Given sepal length (a) {a} cm and petal length (b) {b} cm, it MIGHT NOT BE {species}. <br> <a href='/'>HOME</a>"

if __name__ == '__main__':
    app.run() # domyślnie localhost i port 5000
    