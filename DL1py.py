
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
df.head(n=10)

df.describe()
df.info()
df.isnull().sum()


x = df.drop('medv', axis=1)
x
y = df['medv']
y

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=234)

model1 = StandardScaler()
x_train_scaled = model1.fit_transform(x_train)
x_test_scaled = model1.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout

model2 = Sequential()
model2.add(Dense(512, activation = 'relu', input_dim = 13))
model2.add(Dense(256, activation = 'relu'))
model2.add(Dense(128, activation = 'relu'))
model2.add(Dense(64, activation = 'relu'))
model2.add(Dense(32, activation = 'relu'))
model2.add(Dense(16, activation = 'relu'))
model2.add(Dense(1))
print(model2.summary())

model2.compile(optimizer = 'adam', loss = 'mean_squared_error')

model_fit = model2.fit(x_train_scaled, y_train, epochs = 50, validation split = 0.5, verbose = 1)
res = model2.evaluate(x_test_scaled, y_test)
res

predictions = model2.predict(x_test_scaled)

print("Actual vs Predicted Values:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
print(results)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=results, alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
     
