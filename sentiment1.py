#Sentiment Analysis-getting the sentiment of the tweeter(positive neutral,neagtive)
#from US airline based tweets.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

#text cleansing function
def tweetclean(tweet):
    #html cleansing
    tweet=BeautifulSoup(tweet).get_text()
    
    #remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", tweet)
    
    #lower and split
    words = letters_only.lower().split()                             

    #stopwords
    meanwords=[w for w in words if not w  in stopwords.words("english")]
    
    return(" ".join(meanwords)) 
    


# Importing the training set
dataset_train = pd.read_csv('Tweets.csv')


#Removing the irrelevant data
cols=['airline_sentiment','text']
dataset=dataset_train[cols]

#categorize sentiment
factor=pd.factorize(dataset['airline_sentiment'])
dataset['id']=factor[0]
define=factor[1]

#plotting
fig = plt.figure(figsize=(8,6))
dataset.groupby('id').airline_sentiment.count().plot.bar(ylim=0)
plt.show()

#text cleansing
new_tweet=[]
for i in range(14640):
    new_tweet.append(tweetclean(dataset.text[i]))
data=pd.DataFrame(new_tweet,columns=['text'])

from sklearn.feature_extraction.text import HashingVectorizer
hashing = HashingVectorizer(n_features=2**13)
features=hashing.fit_transform(data.text).toarray()
features=pd.DataFrame(features)

  #more features
new_feature=dataset_train['airline']
new_feature=pd.get_dummies(new_feature,drop_first=True)
new_feature['confidence']=dataset_train['airline_sentiment_confidence']
feature=pd.concat((new_feature,features),axis=1)

  #model
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(feature, dataset['id'], random_state = 0)
model= LinearSVC()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))  

reverse=dict(zip(range(3),define))
y_pred=np.vectorize(reverse.get)(y_pred)  




