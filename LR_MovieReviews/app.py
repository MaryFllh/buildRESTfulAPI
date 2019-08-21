import pickle
import numpy as np

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from model import LRmodel

app = Flask(__name__)
api = Api(app)
model = LRmodel()
clf_path = '/datasets/models/moviewReviewsLogisticRegression.pkl'   

#load trained model
with open(clf_path, 'rb') as f:
    clf = pickle.load(f)
 
#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    
    def get(self):
        
        #use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        prediction = model.predict(np.array([user_query]), clf = clf)
        pred_proba = model.predict_proba(np.array([user_query]), clf = clf)

        #output either 'Negative' or 'Positive' along with the score
        if prediction[0] == 0:
            pred_text = 'Negative'
        
        else:
            pred_text = 'Positive'
        

        #round the predict proba value and set to new variable
        confidence = np.round(pred_proba[0], 3)

        #create JSON object
        output = {'prediction': pred_text, 'confidence': confidence.tolist()}

        return output


#setup the Api resource routing here
#route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug = True)  #set to false when deploying the model
