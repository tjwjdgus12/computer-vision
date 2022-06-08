import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

WIDTH = 272
HEIGHT = 136

# 랜덤시드 고정시키기
np.random.seed(5)
tf.random.set_seed(5)

# 1. 데이터 준비하기
import pandas as pd
dataset = pd.read_csv("3-cushion-helper/ml/data.csv", delimiter=",")
dataset = dataset.values

# 2. 데이터셋 생성하기 : 700
x_train = dataset[:-10,0:6]
y_train = dataset[:-10,6]
x_test = dataset[-10:,0:6]
y_test = dataset[-10:,6]

def getFlipNumber(n):
    d = {0:1,1:0,2:3,3:2,4:4}
    return d[n]

x_train_xflip = x_train.copy()
y_train_xflip = y_train.copy()

x_train_xflip[:, 0] = WIDTH - x_train_xflip[:,0]
x_train_xflip[:, 2] = WIDTH - x_train_xflip[:,2]
x_train_xflip[:, 4] = WIDTH - x_train_xflip[:,4]
for i in range(y_train_xflip.shape[0]):
    y_train_xflip[i] = getFlipNumber(y_train_xflip[i])


x_train_yflip = x_train.copy()
y_train_yflip = y_train.copy()

x_train_yflip[:, 1] = HEIGHT - x_train_yflip[:,1]
x_train_yflip[:, 3] = HEIGHT - x_train_yflip[:,3]
x_train_yflip[:, 5] = HEIGHT - x_train_yflip[:,5]
for i in range(y_train_yflip.shape[0]):
    y_train_yflip[i] = getFlipNumber(y_train_yflip[i])


x_train_flip = x_train.copy()
y_train_flip = y_train.copy()

x_train_flip[:, 0] = WIDTH - x_train_flip[:,0]
x_train_flip[:, 2] = WIDTH - x_train_flip[:,2]
x_train_flip[:, 4] = WIDTH - x_train_flip[:,4]
x_train_flip[:, 1] = HEIGHT - x_train_flip[:,1]
x_train_flip[:, 3] = HEIGHT - x_train_flip[:,3]
x_train_flip[:, 5] = HEIGHT - x_train_flip[:,5]
for i in range(y_train_flip.shape[0]):
    y_train_flip[i] = getFlipNumber(y_train_flip[i])

x_train = np.vstack([x_train, x_train_xflip, x_train_yflip, x_train_flip])
y_train = np.concatenate([y_train, y_train_xflip, y_train_yflip, y_train_flip])

#######################

encoder = LabelEncoder()
encoder.fit(y_train)
y_encoded = to_categorical(encoder.transform(y_train))

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(8, input_dim=6, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.08))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.08))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(5, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 5. 모델 학습시키기
history = model.fit(x_train, y_encoded, epochs=1000, batch_size=16, shuffle=True)
model.summary()

y_loss = history.history['loss']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_acc, c='red', label="Train-set Accuracy")
plt.plot(x_len, y_loss, c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 검증 데이터 확인
pred = model.predict(x_test)
for pre in pred:
    t = [(i, p) for i, p in enumerate(pre)]
    t.sort(key=lambda x: x[1], reverse=True)
    print(t)

# 모델 저장
model.save("3-cushion-helper/temp_model_tttt.h5")