# 순차 데이터 = 텍스트나 시계열 데이터와 같이 순서에 의미가 있는 데이터를 말함
# 텍스트 데이터는 순차 데이터의 필요성이 강조가 되는 데이터
# 앞서 학습시킨 MNIST 데이터는 오히려 섞이고 무분별하게 나눠져야 결과가 더좋은 반면 텍스트 데이터는 순차적인 순서로 나열이 되어야 이전에 입력한 데이터를 기억하는 기능을 사용할 수 있음
# 완전 연결 신경망이나 합성곱 신경망은 앞서 말한 기억하는 기억 장치가 없음
# 하나의 샘플을 사용하여 정방향 계산을 수행하고 나면 그 샘플은 버려지고 다음 샘플을 처리 할 때 재사용하지않음
# 이러한 입력데이터의 흐름이 앞으로만 전달되는 신경망은 피드포워드 신경망이라고함
# 신경망이 이전에 처리했떤 샘플을 다음 샘플을 처리하는데 재사용하기 위해서는 이렇게 데이터 흐름을 앞으로만 전달되어서는 곤란
# 다음 샘플을 위해서 이전 데이터가 신경망 층에 순환될 필요가 있음 
# 이게바로 순환 신경망
# 순환 신경망 = 일반적으로 완전 연결 신경망과 거의 비슷 완전 연결 신경망에 이진 데이터의 처리 흐름을 순환하는 고리 하나만 추가하면됨

# 입력받은 뉴런에서 다음 뉴런으로 OUTPUT을 주되 다시 다음뉴런에서의 값도 같이들고와 저장되는것 
# 이렇게 샘플을 처리하는 한 단계를 타임스텝
# 순환 신경망은 이전 타임스텝의 샘플을 기억하지만 오래된 샘플은 희미해짐
# 순환 신경망에서는 특별히 층을 셀이라고부름 
# 한 셀에는 여러개의 뉴런이 있찌만 완전 연결 신경망과 달리 뉴런을 모두 표시하지않고 하나의 셀로 층을 표현 
# 셀의 출력을 은닉 상태라고부름

# 토큰 = 텍스트에서 공백으로 구분되는 문자열을 뜻함

## 순환 신경망으로 IMDB 리뷰 분류하기

# IMDB 데이터셋 가져오기
#%%
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

# %%
# 훈련 세트와 테스트 세트의 크기를 확인
print(train_input.shape, test_input.shape)

# 훈련세트와 테스트세트 각각 25000개
# %%
# IMDB 리뷰 텍스트는 길이가 제각각 따라서 고정 크기의 2차원 배열에 담기보다는 리뷰마다 별도의 파이썬 리스트로 담아야 메모리 효율적으로 사용가능
# 넘파이 배열은 정수나 실수 이외도 파이썬 배열도 담을 수 있음
# 첫번째 리뷰의 길이 출력
print(len(train_input[0]))

# %%
# 첫 번째 리뷰의 길이는 218개의 토큰으로 이루어져있음 
# 두 번째 리부의 길이 출력
print(len(train_input[1]))
# 하나의 댓글이 하나의 샘플이됨

# %%
# 첫번째 리뷰에 담긴 내용 출력
print(train_input[0])

# IMDB 리뷰 데이터는 이미 정수로 변환되어 있다. 
# %%
# 타깃 데이터 출력
print(train_target[:20])
# %%
# 긍정 혹은 부정으로 판단하기 떄문에 이진분류 문제로 볼수 있으므로 타깃값이 0(부정), 1(긍정)으로 나누어짐
# 훈련 세트에서 검증 세트를 떼어놓기
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input,  train_target, test_size= 0.2, random_state=42)

# %%
# 각 리뷰의 길이를 계산해 넘파이 배열에 담기
import numpy as np
lengths = np.array([len(x) for x in train_input])
# %%
# 넘파이 mean()함수와 median() 함수를 사용해 리뷰 길이의 평균과 중간값 구하기
print(np.mean(lengths), np.median(lengths))

# 리뷰의 평균값은 239 중간값은 178 
# 이 리뷰데이터는 한쪽으로 치우친 분포를 보여줄것

# %%
# 확인
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
# 대부분의 리뷰 길이는 300미만 평균이 중간값보다 높은 이유는 오른쪽 끝에 아주 큰 데이터가 있기 때문

# %%
#  pad_sequences()함수를 이용해 train_input 길이를 100으로 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
# 사용법은 간단 maxlen에 원하는 길이를 지정하면 이보다 긴 경우는 잘라내고 짧은 경우는 0으로 패딩

# %%
# 패딩된 결과 도출
print(train_seq.shape)
# train_input은 파이썬 리스트의 배열이였지만 길이를 100으로 맞춘 train_seq는 이제 2차원 배열이됨

# %%
#train_seq에 있는 첫 번째 배열 출력
print(train_seq[0])

# 샘플에 0패딩값이 없는걸 보아 100보다 길었을것이다 
# %%
# 원본 샘플출력
print(train_input[0][-10:])
# train_seq[0]의 값과 비교하면 정확하게 일치
# 그 말은 즉슨 샘플의 앞부분이 잘렸따는것을 짐작가능
# pad_sequences() 함수는 기본으로 maxlen보다 긴 시퀀스의 앞부분을 자름 
# 이렇게 하는 이유는 일반적으로 시퀀스의 뒷부분이 더 중요하리라 기대하기때문 

# %%
# 여섯번째 샘플 출력
print(train_seq[5])
# 앞부분이 0이 있는걸 보아 이샘플은 100이 안됨
# 같은 이유로 패딩 토큰은 뒷부분이 아니라 앞부분에 추가됨

# %%
# 검증세트의 길이도 100으로 맞추기
val_seq = pad_sequences(val_input, maxlen=100)

# %%
# 순환 신경망 만들기
# 케라스에서는 여러 종류의 순환층 클래스를 제공
# 그중 가장 간단한 것은 SimpleRNN클래스
# 케라스의 Sequential 클래스로 만든 신경망 코드 살펴보기
# Seqiemtial 클래스는 RNN뿐만 아니라 합성곱 신경망이나 일반적인 인공 신경망 모두 만들수 있음 
# 다만 층을 순서대로 쌓기 떄문에 해당 클래스로 이름을 붙임
#%%
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100,500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# %%
train_oh = keras.utils.to_categorical(train_seq)

# %%
# train_seq를 원-핫 인코딩으로 변환하여 train_oh 배열로 만들기
# 배열의 크기 출력
print(train_oh.shape)
# 정수 하나마다 모두 500차원의 배열로 변경되었기 때문에 (20000, 100, 500) 크기의 train_oh로 바뀜

# %%
# 인코딩 잘되었는지 출력
print(train_oh[0][0][:12])
# 처음 12개의 원소를 출력해 보면 열한 번째 원소가 1인 것을 확인가능

# %%
# 나머지원소모두 0인지 확인
print(np.sum(train_oh[0][0]))
# 토근 10이 잘 인코딩됨
# 열한 번째 원소만 1이고 나머지는 모두 0이여서 원-핫 인코딩된 배열의 값을 모두 더한 결과가 1이되었음

# %%
# 같은 방식으로 val_seq도 인코딩하여 바꾸기
val_oh = keras.utils.to_categorical(val_seq)

# %%
# 사용할 훈련 세트와 테스트 세트가 준비됨 
# 사용할 모델의 구조 출력
model.summary()

# SimpleRNN에 전달할 샘플의 크기는 (100, 500)이이지만 이 순환층은 마지막 타임스텝의 은닉상태만 출력
# 이 떄문에 출력 크기가 순환츠으이 뉴런 개수와 동일한 8임을 확인

# %%
# RNN 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

#w 에포크 100 회보다 적은 횟수에서 정확도 대략 80프로이상을 찍음
#%%
# 훈련 손실과 검증 손실을 그래프로 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('eporch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# 훈련 손실은 꾸준히 감소하고 있지만 검증 손실은 대략 스무 번쨰 에포크에서 감소가 둔해지고있다. 
# 적절한 에포크에서 훈련을 멈춤
# 한가지 생각할 점은 이 작업을 하기 위해서 입력 데이터를 원-핫 인코딩으로 변환했음
# 원-핫 인코딩의 입력 데이터가 엄청 커진다는 것
# 실제 train_seq배열과 train_oh 배열의 nbytes 속성을 출력하여 크기를 확인

#%%
print(train_seq.nbytes, train_oh.nbytes)
# 토큰 1개를 500차원으로 늘렸기 떄문에 대략 500배가 커졌음
# 훈련 데이터가 커질수록 더 큰 문제가 될것

# %%
# 단어 임베딩 사용
# 순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법은 단어 임베딩
# 각 단어를 고정된 크기의 실수 벡터로 바꾸어줌
# 원=핫 인코딩보다 보다 저 작은 데이터로 고차원으로 변환
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# Embedding 클래스의 첫 번째 매개변수 500은 어휘 사전의 크기
# 두 번째 매개변수 16은 임베딩 벡터의 크기
# 세번째 input_length 매개변수는 입력 시퀀스의 길이

# %%
# 모델출력
model2.summary()

# %%
# 모델 재훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# 출력 결과를 보면 원-핫 인코딩을 사용한 모델과 비슷한 성능을 보여줌
# 반면에 순환층의 가중치 개수는 훨씬 작고 훈련세트 크기도 훨씬 줄어듬

# %%
# 훈련 손실과 검증 손실  그래프로 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('eporch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# 검증 손실이 더 감소되지 않아 훈련이 적절히 조기 종료됨
# 이에 비해 훈련 손실은 계속 감소함
# 개선의 필요가 있음
# %%
