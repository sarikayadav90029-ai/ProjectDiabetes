import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("diabetes.csv")
print(df.head(10))
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].median())
    X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()

model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train,y_train,epochs=50,batch_size=32)
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

joblib.dump(scaler, "scaler.pkl")
model.save("diabetes_model.keras")