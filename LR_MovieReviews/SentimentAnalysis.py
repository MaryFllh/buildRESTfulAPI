import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class LRmodel():



    def train(self, clf,X, y):
        """
        Train the model
        """
        
        clf.fit(X, y)       
     

    def pickle_clf(self, clf,path ='/datasets/models/moviewReviewsLogisticRegression.pkl'):
       
        with open(path, 'wb') as f:
            pickle.dump(clf, f)
            print("Pickled classifier at {}".format(path))
    

    def predict(self, X, clf):
        """
        Predict the tag of the input comment 
        """
        
      
        y_pred = clf.predict(X)
       
        return y_pred
    
    def predict_proba(self, X, clf):

        """
        Return the confidence of each comment being relevent to it 
        """
        
        y_proba = clf.predict_proba(X)
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



