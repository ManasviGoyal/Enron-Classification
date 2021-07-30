#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from time import perf_counter
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv(r'C:\Users\mgman\Downloads\Emails\spam_ham_dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace = True)
df.columns = ['Label', 'Text', 'Label_Number']
#df = pd.read_excel(r'C:\Users\mgman\Downloads\Emails\All_Emails.xlsx', sheet_name='Sheet1')
#df.drop('ID', axis=1, inplace = True)
df.head()

df.shape

df.info()

df.isna().sum()

df['Label_Number'].value_counts()

plt.figure(figsize = (8, 6))
sns.countplot(data = df, x = 'Label');

def count_words(text):
    words = word_tokenize(text)
    return len(words)
df['count']=df['Text'].apply(count_words)
df['count']

df.groupby('Label_Number')['count'].mean()

get_ipython().run_cell_magic('time', '', 'def clean_str(string, reg = RegexpTokenizer(r\'[a-z]+\')):\n    # Clean a string with RegexpTokenizer\n    string = string.lower()\n    tokens = reg.tokenize(string)\n    return " ".join(tokens)\n\nprint(\'Before cleaning:\')\ndf.head()')

print('After cleaning:')
df['Text'] = df['Text'].apply(lambda string: clean_str(string))
df.head()

df["Text"] = [' '.join([item for item in x.split() 
                  if item not in 'subject']) 
                  for x in df["Text"]]
df.head()

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stemming (text):
    return ''.join([stemmer.stem(word) for word in text])
df['Text']=df['Text'].apply(stemming)
df.head()

X = df.loc[:, 'Text']
y = df.loc[:, 'Label_Number']

print(f"Shape of X: {X.shape}\nshape of y: {y.shape}")

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df.Text)
y = df.Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
models = {
    "Random Forest": {"model":RandomForestClassifier(), "perf":0},
    "Gradient Boosting": {"model":GradientBoostingClassifier(), "perf":0},
    "MultinomialNB": {"model":MultinomialNB(), "perf":0},
    "Logistic Regr.": {"model":LogisticRegression(), "perf":0},
    "KNN": {"model":KNeighborsClassifier(), "perf":0},
    "Decision Tree": {"model":DecisionTreeClassifier(), "perf":0},
    "SVM (Linear)": {"model":LinearSVC(), "perf":0},
    "SVM (RBF)": {"model":SVC(), "perf":0}
}

for name, model in models.items():
    start = perf_counter()
    model['model'].fit(X_train, y_train)
    duration = perf_counter() - start
    duration = round(duration,2)
    model["perf"] = duration
    print(f"{name:20} trained in {duration} sec")

models_accuracy = []
for name, model in models.items():
    models_accuracy.append([name, model["model"].score(X_test, y_test),model["perf"]])
models_accuracy1 = []
for name, model in models.items():
    models_accuracy1.append([name, model["model"].score(X_train, y_train),model["perf"]])

df_accuracy = pd.DataFrame(models_accuracy)
df_accuracy.columns = ['Model', 'Test Accuracy', 'Training time (sec)']
df_accuracy.sort_values(by = 'Test Accuracy', ascending = False, inplace=True)
df_accuracy.reset_index(drop = True, inplace=True)
df_accuracy

plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y ='Test Accuracy', data = df_accuracy)
plt.title('Accuracy on the test set\n', fontsize = 15)
plt.ylim(0.825,1)
plt.show()

plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y = 'Training time (sec)', data = df_accuracy)
plt.title('Training time for each model in sec', fontsize = 15)
plt.ylim(0,20)
plt.show()


# MutinomialNB gives the best results in terms of both the training tone and the Test Accuracy

# In[23]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
parameters = {"alpha": [0.2,1,2,5,10], "fit_prior": [True, False]}

grid = GridSearchCV(MultinomialNB(), param_grid=parameters)
grid.fit(X_train,y_train)

# Create a DataFrame with the best Hyperparameters
pd.DataFrame(grid.cv_results_)[['params','mean_test_score']]\.sort_values(by="mean_test_score", ascending=False)

grid.best_params_

from sklearn.naive_bayes import MultinomialNB
alpha, fit_prior = grid.best_params_['alpha'], grid.best_params_['fit_prior']
model = MultinomialNB(alpha = alpha)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(f'## Accuracy: {round(accuracy_score(y_test,y_pred),3)*100}%\n')

print(classification_report(y_test,y_pred))

def display_result(df, number=2):
    for i in range(number):
        msg = df['Text'].iloc[i]
        label = df["Label"].iloc[i]
        msg_vec = cv.transform([msg])
        pred_label = model.predict(msg_vec)
        print(f"**Real: {label}, Predicted: {pred_label[0]}**")
        print(f"**E-Mail:** {msg}")
        print("_______________________________________________________________")
    
df_spam = df[df['Label'] == 'spam']
df_ham = df[df['Label'] == 'ham']
display_result(df_spam)
display_result(df_ham)
