import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import itertools as it

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 3

# 랜덤시드 고정시키기
np.random.seed(5)
tf.random.set_seed(5)

print("Data loading...")

# 1. 데이터 준비하기
import pandas as pd
dataset = pd.read_csv("data.csv", delimiter=",")
dataset = dataset.values

# 2. 데이터셋 생성하기 : 700
x_train = dataset[:200,0:6]
y_train = dataset[:200,6]
x_test = dataset[200:,0:6]
y_test = dataset[200:,6]

print("Data expanding...")

def getFlipNumber(n):
    d = {0:1,1:0,2:3,3:2,4:4}
    return d[n]


x_train_copy = x_train.copy()
y_train_copy = y_train.copy()

x_train_iter = x_train_copy.copy()
y_train_iter = y_train_copy.copy()

# x_temp = []
# y_temp = []

# from tqdm import tqdm
# for k, x in enumerate(x_train_copy):
#     iteration = it.product([-1,0,1],repeat=6)
#     for iters in tqdm(iteration):
#         x_temp.append([x[i]])

#         new_x = x.copy()

#         for i, iter in enumerate(iters):
#             new_x[i] = new_x[i] + iter

#         x_train_iter= np.vstack([x_train_iter,new_x])
#         y_train_iter= np.append(y_train_iter,y_train_iter[k])


x_train = x_train_iter.copy()
y_train = y_train_iter.copy()

x_train_iter_copy = x_train_iter.copy()
y_train_iter_copy = y_train_iter.copy()

print("Data expanding...")

x_train_xflip = x_train_iter_copy.copy()
y_train_xflip = y_train_iter_copy.copy()

x_train_xflip[:, 0] = WIDTH - x_train_xflip[:,0]
x_train_xflip[:, 2] = WIDTH - x_train_xflip[:,2]
x_train_xflip[:, 4] = WIDTH - x_train_xflip[:,4]
for i in range(y_train_xflip.shape[0]):
    y_train_xflip[i] = getFlipNumber(y_train_xflip[i])


x_train_yflip = x_train_iter_copy.copy()
y_train_yflip = y_train_iter_copy.copy()

x_train_yflip[:, 1] = HEIGHT - x_train_yflip[:,1]
x_train_yflip[:, 3] = HEIGHT - x_train_yflip[:,3]
x_train_yflip[:, 5] = HEIGHT - x_train_yflip[:,5]
for i in range(y_train_yflip.shape[0]):
    y_train_yflip[i] = getFlipNumber(y_train_yflip[i])



x_train_flip = x_train_iter_copy.copy()
y_train_flip = y_train_iter_copy.copy()

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

print("Data expanding complete")

#######################



encoder = LabelEncoder()
encoder.fit(y_train)
y_encoded = to_categorical(encoder.transform(y_train))

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(16, input_dim=6, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))

model.add(Dense(5, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 5. 모델 학습시키기
model.fit(x_train, y_encoded, epochs=1500, batch_size=32)
model.summary()

# 6. 모델 평가하기
# scores = model.evaluate(x_test, y_test)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

def getAccuracy(arr):
    hit = 0
    for i,elem in enumerate(arr):
        print(i+1, x_test[i], y_test[i], elem)
        if y_train[i] == elem:
            hit+=1
    return hit/len(arr)

pred = model.predict(x_test)
for pre in pred:
    t = [(i, p) for i, p in enumerate(pre)]
    t.sort(key=lambda x: x[1], reverse=True)
    print(t)

model.save("ml_model")