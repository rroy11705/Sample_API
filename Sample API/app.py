from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
from model import KNeighbor_Model
import numpy as np

app = Flask(__name__)
api = Api(app)

model = KNeighbor_Model()

path  = 'Lib/Models/Survival.pkl'

with open(path, 'rb') as f:
    model.clf = pickle.load(f) 

parser = reqparse.RequestParser()
parser.add_argument('Pclass', type = int)
parser.add_argument('Age', type = float)
parser.add_argument('SibSp', type = int)
parser.add_argument('Parch', type = int)
parser.add_argument('Fare', type = float)
parser.add_argument('Sex_female', type = int)
parser.add_argument('Sex_male', type = int)
parser.add_argument('Embarked_C', type = int)
parser.add_argument('Embarked_Q', type = int)
parser.add_argument('Embarked_S', type = int)


class Survival(Resource):
    def get(self):
        args = parser.parse_args()
        

        predicted = model.predict(args)
        confidence = model.predict_proba(args)

        if predicted == 1:
            pred_text = "Survived"
        elif predicted == 0:
            pred_text = "Died"

        output = {'Prediction' : pred_text, 'Confidence' : confidence}

        return output

api.add_resource(Survival, '/')

if __name__ == '__main__':
    app.run(debug=True)