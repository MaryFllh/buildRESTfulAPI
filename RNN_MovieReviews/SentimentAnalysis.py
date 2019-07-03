import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import html
import nltk
import itertools
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from lxml.etree import tostring
import lxml
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

class RNNmodel():

    def __init__(self):
        
        self.vocab_size =  1000
        self.comment_len =  50
        self.epochs = 32
        self.emb_dim = 128
        self.batch_size = 32


    def preprocessing(self, text):
        
        """
        input: the raw comment
        output: cleaned and processed comment
        """
        text = text[0]
        symbol_pattern = re.compile('[/(){}\[\]\|@,;]')
        all_other_symbols = re.compile('[^0-9a-z #+_]')
        remove_stopwords = set(stopwords.words('english'))
        text = BeautifulSoup(text, "lxml").text # HTML decoding
        text = text.lower() # lowercase the text
        text = symbol_pattern.sub(' ', text)   #replace symbols in symbol_pattern by space in text
        text = all_other_symbols.sub('', text) # replace non (alpha)numeric characters from the text with space 
        text = ' '.join(word for word in text.split() if word not in remove_stopwords) # delete stopwors from text
        
        return text

    def tokenize(self, text):
        """
        Given vocaulary size and length of each comment to consider, tokenize each input text
        """

        tokenizer = Tokenizer(num_words = self.vocab_size, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        tokenized = pad_sequences(sequences, maxlen = self.comment_len)
        print('tokenised shape is: ', tokenized.shape)

        return tokenized

    def RNN(self):
        """
        Define the neural network architecture
        """
        model = Sequential()
        X = self.tokenize()
        model.add(Embedding(self.vocab_size, self.emb_dim, input_length = X.shape[1])) #Embedding layer
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(256, dropout =0.2, recurrent_dropout = 0.2)) #recurrent layer
        model.add(Dense(20, activation ='softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    def train(self, X, y):
        """
        Train the model
        """
        clf = self.RNN()
        clf.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, validation_split = 0.2,callbacks =[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])       
        self.clf = clf 

    def pickle_clf(self, path ='/home/fallahm/Documents/datasets/models/SentimentClassifier.pkl'):
       
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))
    

    def predict(self, X, clf):
        """
        Predict the tag of the input comment 
        """
        
      
        y_pred = clf.predict_classes(X, batch_size = 32)
       
        return y_pred
    
    def predict_proba(self, X, clf):

        """
        Return the confidence of each comment being relevent to it 
        """
        
        y_proba = clf.predict(X, batch_size = 32)
        return y_proba


    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title= 'Confusion Matrix',
                          cmap=plt.cm.Blues):
        """
        cm : confusion matrix to be plotted
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)
        
        # Only use the labels that appear in the data
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                        horizontalalignment = "center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()



