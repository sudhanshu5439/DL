#5b) Aim: Evaluating feed forward deep network for multiclass Classification using KFold cross-validation.

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Flower.csv', header=None)

# Split the dataset into input and output variables
X = df.iloc[:, 0:4].astype(float)
y = df.iloc[:, 4]

# Encode the string output into numeric output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_Y = np_utils.to_categorical(encoded_y)

# Define the baseline model
def baseline_model():
    # Create the model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the estimator for KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=10)

# Perform K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)

# Print the cross-validation results
print("Accuracy: %.2f%%" % (results.mean() * 100))
