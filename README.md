# Build RESTful API
This is a step by step tutorial to build a RESTful API from a machine learning model using Flask and its deployment using AWS.\
I have used two datasets:  

1- Stackoverflow questions: 40000 tagged questions from 20 different categories such as c#, python, .net, and much more. An RNN was trained on this dataset to predict(suggest) tags for a user's query. Since there are many classes and not enough data the model's accuracy is relatively low: 77.3%  
2- Movie Reviews: 16278 movie reviews with 5 star ratings, for simplicity I only focused on 1 and 5 star ratings and consider them as negative and positive ratings respectively. An RNN and Logistic regression model was trained on this dataset. As there is not a lot of training data available the Logistic regression model outperforms the RNN with 92.78% and 87.4% accuracy respectively.  

Codes for all three models are uploaded, but I only deployed the sentiment analysis for movie reviews with the LR model as it gained 
higher accuracy.  

# Steps for testing the API
You can either use the trained model in the models folder and run the app.py script or train the model by running buildModel.py   
1- After training the model (running buildModel.py) or just loading the trained model (in app.py script), run app.py. Make sure that the pickle model is in the path.  
2- From another terminal make a GET request with a query, e.g: http http://127.0.0.1:5000/ query == "The movie was extraordinary!"  
3- The output should be similar to below:    
HTTP/1.0 200 OK  
Content-Length: 85  
Content-Type: application/json  
Date: Wed, 03 Jul 2019 19:10:11 GMT  
Server: Werkzeug/0.15.4 Python/3.6.7  

{  
    "confidence": [  
        0.0,  
        1.0  
    ],  
    "prediction": "Positive"  
}   
# Next Steps (coming soon):  
Deploying the API.

# Resources:
https://towardsdatascience.com/auto-tagging-stack-overflow-questions-5426af692904
https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
