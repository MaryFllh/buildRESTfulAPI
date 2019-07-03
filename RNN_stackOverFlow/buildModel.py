from SentimentAnalysis import RNNmodel
import pandas as pd 
from sklearn.model_selection import train_test_split

def buildModel():
    """
    If you want to train the model from scratch, run this script. It will load the dataset, select the features and labels amd split into train and test, train and eventually save into pickle
    """
    model = RNNmodel()
    data = pd.read_csv('/datasets/stack-overflow-data.csv')
    data = data[pd.notnull(data['tags'])]  # we only consider the rows with availbale tags
    data['post'] = data['post'].apply(model.preprocessing())
     
    #map each tag to a number
    data.loc[data['tags'] == 'angularjs', 'labels'] = 0
    data.loc[data['tags'] == 'c#', 'labels'] = 1
    data.loc[data['tags'] == 'python', 'labels'] = 2
    data.loc[data['tags'] == 'c', 'labels'] = 3
    data.loc[data['tags'] == 'ios', 'labels'] = 4
    data.loc[data['tags'] == 'java', 'labels'] = 5
    data.loc[data['tags'] == '.net', 'labels'] = 6
    data.loc[data['tags'] == 'jquery', 'labels'] = 7
    data.loc[data['tags'] == 'javascript', 'labels'] = 8
    data.loc[data['tags'] == 'sql', 'labels'] = 9
    data.loc[data['tags'] == 'html', 'labels'] = 10
    data.loc[data['tags'] == 'c++', 'labels'] = 11
    data.loc[data['tags'] == 'php', 'labels'] = 12
    data.loc[data['tags'] == 'ruby-on-rails', 'labels'] = 13
    data.loc[data['tags'] == 'css', 'labels'] = 14
    data.loc[data['tags'] == 'iphone', 'labels'] = 15
    data.loc[data['tags'] == 'objective-c', 'labels'] = 16
    data.loc[data['tags'] == 'asp.net', 'labels'] = 17
    data.loc[data['tags'] == 'android', 'labels'] = 18
    data.loc[data['tags'] == 'mysql', 'labels'] = 19

    labels = to_categorical(data['labels'], num_classes=20) #one-hot encoded tags
    X = model.tokenize(data['post'])
    X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)
    model.train(X_train, y_train)
    model.pickle_clf()
    
    if __name__ == "__main__":
        build_model()