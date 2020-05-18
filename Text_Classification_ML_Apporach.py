# Text Classification by converting the text to Tfidf vectors
# We will be working on the Amazon reviews a dataset from Kaggle competitions which is zipped in a bz2 format.

#Import the required packages to unzip the file
import bz2

# Read the file into an object after extraction
file = bz2.BZ2File('train.ft.txt.bz2')

# Read the file contents
file_lines = file.readlines() 
file_lines = [x.decode('utf-8') for x in file_lines]


#Extract the Label and Revieiws from the text file by perfoming text split
labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in file_lines]
reviews = [x.split(' ', 1)[1][:-1].lower() for x in file_lines]

# Import Regular Expressions(re) package to preprocess the reviews and remove unnecessary text parts
import re
for i in range(len(reviews)):
    reviews[i] = re.sub('\d','0',reviews[i])
for i in range(len(reviews)):
    if 'www.' in reviews[i] or 'http:' in reviews[i] or 'https:' in reviews[i] or '.com' in reviews[i]:
        reviews[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", reviews[i])

#Import pandas and generate DataFrame of the Revieiws and Labels
import pandas as pd
reviews_df = pd.DataFrame.from_dict({'Reviews':reviews,'labels':labels})

#Check for any missing elements in Review
print(reviews_df.isna().sum())

#Check for any empty '' strings in the reviews
blanks = []
for idx,rev,lbl in reviews_df.itertuples():
    if type(rev) == str:
        if rev.isspace():
            blanks.append(idx)

print(len(blank))

# There are no missing or empty string elements. Let's go ahead and split the dataset into train and test sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(reviews_df['reviews'],reviews_df['labels'],test_size=.33,random_state=42)

#Now let us conver the reviews to Tfidf Vectors
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95,min_df=0.12,stop_words='english')
X_train = tfidf.fit_transform(X_train)

X_test = tfidf.fit_transform(X_test)

#Check the number of elements in each label class
print(reviews_df['labels'].value_counts())

#First let us try logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

predicted = lr.predict(X_test)

#Now let us check the performance of Logistic Regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#First let us build the confusion matrix to check TP,FP,FN,TN values
print(confusion_matrix(y_true=y_test,y_pred=predicted))
#Now the accuracy of the model:
print(f"Accuracy: {accuracy_score(y_true=y_test,y_pred=predicted)})
# Classification Report to check Presision, Recall and F-1 scores
print(classification_report(y_true=y_test,y_pred=predicted))

#Now let us try Linear Support Vector Classier
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(X_train,y_train)

predicted = svm.predict(X_test)

# Check the accuracy of the model
print(f"Accuracy:{accuracy_score(y_test,predicted)})

#Finally let's try K-Nearest Neighbor and predict the last 10 values in test set.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

predicted = knn.predict(X_test[:10])


# Print the accuracy of KNN model:
print(f"Accuracy:{accuracy_score(y_test[:10],predicted)}")
