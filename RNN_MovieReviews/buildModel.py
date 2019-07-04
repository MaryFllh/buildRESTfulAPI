from model import RNNmodel
import pandas as pd 
from sklearn.model_selection import train_test_split

def buildModel():
    """
    If you want to train the model from scratch, run this script. It will load the dataset, select the features and labels amd split into train and test, train and eventually save into pickle
    """
    model = RNNmodel()
    data = pd.read_csv('/datasets/stack-overflow-data.csv')

    data = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]  #select only 0 and 4 star ratings as positive or negative
    data['Binary'] = data.apply(lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)
    data['Phrase'] = data['Phrase'].apply(model.preprocessing()) 
    labels = to_categorical(data['Binary'], num_classes=2)
    
    X = model.tokenize(data['Phrase'])   #extract features
    X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)
    model.train(X_train, y_train)
    model.pickle_clf()
    
    if __name__ == "__main__":
        build_model()
