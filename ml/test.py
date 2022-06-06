import numpy as np

from keras.models import Sequential
from keras.layers import Dense



# 랜덤시드 고정시키기
import pandas as pd
np.random.seed(5)

# 1. 데이터 준비하기
dataset = pd.read_csv("diabetes.csv", delimiter=",")
dataset = dataset.values

# 2. 데이터셋 생성하기 : 700
x_train = dataset[:700,0:8]
y_train = dataset[:700,8]
x_test = dataset[700:,0:8]
y_test = dataset[700:,8]

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=1500, batch_size=64)

# 6. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))