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
clf_path = '/datasets/models/stackOverflowRNN.pkl'

#load the trained model
with open(clf_path, 'rb') as f:
    clf = pickle.load(f)
    graph = tf.get_default_graph()
 

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    
    def get(self):
        
        #use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
    
        # preprocess and tokenize the user's query and make a prediction
        #import pdb; pdb.set_trace()
        uq_preprocessed = model.preprocessing(np.array([user_query]))
        uq_tokenized = model.tokenize(np.array([uq_preprocessed]))
        with graph.as_default():
        #[X_test[0].reshape(1,X_test[0].shape[0])]
            prediction = model.predict(uq_tokenized, clf = clf)
            pred_proba = model.predict_proba(uq_tokenized, clf = clf)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction[0] == 0:
            pred_text = 'angularjs'
        
        elif prediction[0] == 1:
            pred_text = 'c#'
        
        elif prediction[0] == 2:
            pred_text = 'python'

        elif prediction[0] == 3:
            pred_text = 'c'
        
        elif prediction[0] == 4:
            pred_text = 'ios'
        
        elif prediction[0] == 5:
            pred_text = 'java'
        
        elif prediction[0] == 6:
            pred_text = '.net'
        
        elif prediction[0] == 7:
            pred_text = 'jquery'
        
        elif prediction[0] == 8:
            pred_text = 'javascript'
        
        elif prediction[0] == 9:
            pred_text = 'sql'

        elif prediction[0] == 10:
            pred_text = 'html'

        elif prediction[0] == 11:
            pred_text = 'c++'
        
        elif prediction[0] == 12:
            pred_text = 'php'
        
        elif prediction[0] == 13:
            pred_text = 'ruby-on-rails'
        
        elif prediction[0] == 14:
            pred_text = 'css'

        elif prediction[0] == 15:
            pred_text = 'iphone'

        elif prediction[0] == 16:
            pred_text = 'objective-c'
        
        elif prediction[0] == 17:
            pred_text = 'asp.net'
        
        elif prediction[0] == 18:
            pred_text = 'android'
        
        elif prediction[0] == 19:
            pred_text = 'mysql'
        
        


        # round the predict proba value and set to new variable
        confidence = np.round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence.tolist()}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
