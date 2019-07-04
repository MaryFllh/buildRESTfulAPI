from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import RNNmodel
import keras
import tensorflow as tf
global graph

keras.backend.clear_session()
app = Flask(__name__)
api = Api(app)
model = RNNmodel()
clf_path = '/datasets/models/movieReviewsRNN.pkl'

#load the trained model
with open(clf_path, 'rb') as f:
    clf = pickle.load(f)
    graph = tf.get_default_graph()  #set this to default graph
 

#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    
    def get(self):
        
        #use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
    
        #preprocess and tokenize the user's query and make a prediction
        uq_preprocessed = model.preprocessing(np.array([user_query]))
        uq_tokenized = model.tokenize(np.array([uq_preprocessed]))
        with graph.as_default():

            prediction = model.predict(uq_tokenized, clf = clf)
            pred_proba = model.predict_proba(uq_tokenized, clf = clf)

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
    app.run(debug=True)
