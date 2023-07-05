#4-b)  Aim: Using a deep field forward network with two hidden layers for performing classification and predicting the probability of class.

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=500)

Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
Ynew_prob = model.predict(Xnew)
Ynew_class = [1 if p >= 0.5 else 0 for p in Ynew_prob]

for i in range(len(Xnew)):
    print("X=%s, Predicted probability=%s, Predicted class=%s" % (Xnew[i], Ynew_prob[i], Ynew_class[i]))

