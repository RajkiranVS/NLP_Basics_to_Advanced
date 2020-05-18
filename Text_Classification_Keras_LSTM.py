
#Let's import the required packages
import bz2
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gc

# Let's read the zipped file and try to read lines in the zipped text file

file = bz2.BZ2File('train.ft.txt.bz2')
file_lines = file.readlines()
file_lines = [line.decode('utf-8') for line in file_lines]

#Now split the label and reviews part from the file
label = [0 if y.split(' ')[0] == '__label__1' else 1 for y in file_lines]

review = [review.split(' ', 1)[1][:-1].lower() for review in file_lines]
# Finally delte the support variables as they occupy space in memory and thus casuing hinderence in the capacity of usage for our model
del(file,file_lines)
gc.collect()


# Before we proceed further, let us create the DataFrame of reviews(X) and labels(y)
reviews = pd.DataFrame.from_dict({'reviews':review,'label':label})


# Let us check number of values belonging to each class
reviews['label'].value_counts()

# Clean the reviews from unnecessary texts
reviews['reviews'] = reviews['reviews'].apply(lambda rev: re.sub('\d','0',rev))
reviews['reviews'] = reviews['reviews'].apply(lambda rev: re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", rev))

# Create the parameters that will be used by the Tokenizer and our model
max_features = 20000
maxlen = 100

#Call the Tokenizer module
tkz = Tokenizer(num_words=max_features)


#Fit the Tokenizer on the texts of reviews column
tkz.fit_on_texts(reviews['reviews'])

# Generate the sequence from the tokenized reviews
tokenized_review = tkz.texts_to_sequences(reviews['reviews'])
X = pad_sequences(tokenized_review, maxlen=maxlen)

# Let's have a glance at the first sequence
print(X[0])

# Split the data into train(alloting 33% for test) and text to evaluate the model's performance
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,reviews['label'],test_size=.33,random_state=42)

# Let us check number of rows in train and test sets
print(f"Number of elements in train: {len(X_train)}")
print(f"Number of elements in test: {len(X_test)}")


# Import the required modules for model building
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

# Create the module that builds the model 
def create_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 25, input_length=maxlen))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))

    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model


# Let us now create the model and check the summary of the network
model = create_model(max_features,maxlen)

# Set the model parameters
batch_size = 2048
epochs = 7

# Import required support packages like callback for the model
from keras.callbacks import EarlyStopping, ModelCheckpoint
weight_path="early_weights.hdf5"
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Here early stopping works on the performance of validation loss. If the validation loss decrease in an epoch compared to previous, the model is saved as hdf5 file
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
callbacks = [checkpoint, early_stopping]

#As we have used softmax as activation for last layer, we need to one-hot encode our label/target/y variable.
from keras.utils import to_categorical
y_train = to_categorical(y_train)

#Finally we are ready to train our model. Relax it would take several hours to complete the epochs. For me each epoch took 1.5 hours
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle = True, validation_split=0.20, callbacks=callbacks)


#Let's call the load_model module from Keras.models and load our early stopped checkpoint
from keras.models import load_model
model = load_model('early_weights.hdf5')

#Let's try to predict the values for last 10 values
predictions = model.predict_classes(X_test[:10])

#Print the last 10 predictions
print(f" Predicted last 10: {predictions}")

#Finally we can check the accuracy thus performance of our model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(confusion_matrix(y_test[:10],predictions))
print(f"Accuracy: {accuracy_score(y_test[:10],predictions)}")
print(classification_report(y_test[:10],predictions))
