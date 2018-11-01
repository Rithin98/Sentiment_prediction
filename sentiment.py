#Sentiment Analysis-getting the sentiment of the tweeter(positive neutral,neagtive)
#from US airline based tweets.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


#text to features
#usingTfidvectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#features = tfidf.fit_transform(dataset.text).toarray()
#features=pd.DataFrame(features)

#using hashingvectorizer
from dask.distributed import Client
from sklearn.externals.joblib import parallel_backend
client=Client(processes=False)
with parallel_backend('dask'):    
  from sklearn.feature_extraction.text import HashingVectorizer
  hashing = HashingVectorizer(n_features=2**13)
  features=hashing.fit_transform(dataset.text).toarray()
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
  
 #deep learning 
  
from keras.models import Sequential
classifier=Sequential()
from keras.layers import Dense,Dropout
#first layer
classifier.add(Dense(units=10,activation='relu',kernel_initializer='glorot_uniform',input_dim=8198))
classifier.add(Dropout(p=0.3))

#second layer
classifier.add(Dense(units=25,activation='relu',init='glorot_uniform'))
classifier.add(Dropout(p=0.3))

from keras.utils import to_categorical
y_binary = to_categorical(y_train)

#output layer
classifier.add(Dense(units=3,activation='softmax',init='glorot_uniform'))

#compiling and fitting
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_binary,epochs=25,batch_size=32)

#predict
y_pred2=classifier.predict(X_test)
y_pred2=y_pred2.argmax(axis=1)

from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
random_classifier = RandomForestClassifier(n_estimators=55)
#parameters = { 'max_features':np.arange(5,10),'n_estimators':[500],'min_samples_leaf': [10,50,100,200,500]}
#random_grid = GridSearchCV(random_classifier, parameters, cv = 5,scoring='accuracy')
random_classifier.fit(X_train, y_train)
y_pred3=random_classifier.predict(X_test)

#using xgb classifier  
from xgboost import XGBClassifier
model1= XGBClassifier(min_child_weight=5,max_depth=3 )
model1.fit(X_train, y_train)
y_pred1=model1.predict(X_test) 
  

#description of the prediction
#from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#conf_mat = confusion_matrix(y_test, y_pred)
#acc=accuracy_score(y_test, y_pred)
reverse=dict(zip(range(3),define))
#print(conf_mat)
#print(acc)
#print(classification_report(y_test, y_pred, target_names=dataset['airline_sentiment'].unique()))
y_test=np.vectorize(reverse.get)(y_test)
y_pred=np.vectorize(reverse.get)(y_pred)
y_pred1=np.vectorize(reverse.get)(y_pred1)
y_pred2=np.vectorize(reverse.get)(y_pred2)
y_pred3=np.vectorize(reverse.get)(y_pred3)




