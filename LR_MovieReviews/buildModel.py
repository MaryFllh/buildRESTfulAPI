from SentimentAnalysis import LRmodel
import pandas as pd 
from sklearn.model_selection import train_test_split

def buildModel():
	"""
	If you want to train the model from scratch, run this script. It will load the dataset, select the features and labels amd split into train and test, train and eventually save into pickle
	"""
    
    model = LRmodel()
    data = pd.read_csv('/datasets/stack-overflow-data.csv')   #load the data
    data = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]
    data['Binary'] = data.apply(lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)
	
	X = data.Phrase
    labels = data.Binary
    X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)
	clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
    model.train(clf = clf, X = X_train, y = y_train)
    model.pickle_clf()
    
    if __name__ == "__main__":
        build_model()