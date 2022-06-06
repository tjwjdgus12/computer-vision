from hashlib import new
import tensorflow as tf
from tensorflow import keras

import itertools as it


import numpy as np
import matplotlib.pyplot as plt
 
WIDTH = 816
HEIGHT = 408
BALL_RADIUS = 10

def getAccuracy(arr):
    hit = 0

    for i,elem in enumerate(arr):
        if y_train[i] == elem:
            hit+=1

    return hit/len(arr)

def getFlipNumber(n):
    if n in [1,2,3]:
        return n+3
    if n in [4,5,6]:
        return n-3
    if n in [7,8,9]:
        return n+3
    if n in [10,11,12]:
        return n-3
    
    return n

tf.random.set_seed(0)

data = np.genfromtxt('data.csv',delimiter=',')

x_train = data[:,0:-1]
y_train = data[:,-1]

# print(x_train)
# print(y_train)

# print(x_train.shape)
# print(y_train.shape)

x_train_copy = data[:,0:-1].copy()
y_train_copy = data[:,-1].copy()

x_train_iter = x_train_copy.copy()
y_train_iter = y_train_copy.copy()



for k, x in enumerate(x_train_copy):
    iteration = it.product([-1,0,1],repeat=6)
    for iters in iteration:
        new_x = x.copy()

        for i, iter in enumerate(iters):
            new_x[i] = new_x[i] + iter

        x_train_iter= np.vstack([x_train_iter,new_x])
        y_train_iter= np.append(y_train_iter,y_train_iter[k])



print("s1",x_train_iter.shape)
print("s2",y_train_iter.shape)

x_train = x_train_iter.copy()
y_train = y_train_iter.copy()

x_train_iter_copy = x_train_iter.copy()
y_train_iter_copy = y_train_iter.copy()


# x_train_xflip = x_train_copy.copy()
# y_train_xflip = y_train_copy.copy()

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

print(x_train.shape)
print(y_train.shape)

# 2. 뉴런층 만들기
input_layer = tf.keras.layers.InputLayer(input_shape=(6,))
hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
output_layer = tf.keras.layers.Dense(units=12, activation='softmax')


# 3. 모델 구성하기
model = tf.keras.Sequential([
  input_layer,
  hidden_layer,
  output_layer
  ])


# 4. 모델 컴파일하기
model.compile(loss='mse', optimizer='Adam')


# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 5. 은닉층의 출력 확인하기
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
intermediate_output = intermediate_layer_model(x_train)

print('======== Inputs ========')
print(x_train)

print('\n======== Weights of Hidden Layer ========')
print(hidden_layer.get_weights()[0])

print('\n======== Outputs of Hidden Layer ========')
print(intermediate_output)



# 6. 출력층의 출력 확인하기
pred = model.predict(x_train)

print('\n======== Outputs of Output Layer ========')
print(pred)
print(pred.shape)

res = [ np.argmax(p) for p in pred ]
print(res)
# print(y_train)
# for y in y_train:
#     print(y)
print( getAccuracy(res) )