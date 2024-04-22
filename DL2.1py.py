
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("2_letter_recognition.data", sep = ",")

X = dataset.iloc[:, 1 : 17]
Y = dataset.select_dtypes(include = [object])

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 10)

scaler = StandardScaler()
scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
mlp = MLPClassifier(hidden_layer_sizes = (250, 300), max_iter = 1000000, activation = 'logistic')
# !pip install yellowbrick
from yellowbrick.classifier import confusion_matrix
cm = confusion_matrix(mlp,X_train,Y_train.values.ravel(), X_validation, Y_validation.values.ravel(), classes="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(','))

cm.fit(X_train, Y_train.values.ravel())

cm.score(X_validation, Y_validation)
predictions = cm.predict(X_validation)
predictions

print("Accuracy: ", accuracy_score(Y_validation, predictions))

 