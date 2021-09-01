# 고급 순환층 LSTM과 GRU 
# 이두개의 층은 SimpleRNN보다 계산이 훨씬 복잡하다. 하지만 성능이 뛰어나기 떄문에 신경망에 많이 채택되고있음

# LSTM = Long Short-Term Memory의 약자이고 말그대로 단기 기억을 기억하기 위해 고안
# 구조가 복합함
# 입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러개 가지고있음
# 은닉 상태는 이전 타임스텝의 은닉상태를 가중치에 곱한 후 활성화 함수를 통과시켜 다음 은닉 상태를 만듬 
# 이떄 기본 순환층과는 달리 시그모이드 활성화 함수를 사용 

# 은닉 상태관련은 검색 권고

# LSTM 신경망 훈련하기
#%%
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input,val_input, train_target, val_target =train_test_split(train_input,train_target, test_size=0.2, random_state=42)

# %%
# 케라스의 pad_sequences()함수로 각 샘플의 길이를 100에 맞추고 부족할 때는 패딩을 추가
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

# %%
# LSTM 셀을 사용한 순환층을 제작 SimpleRNN 클래스를 LSTM클래스로 변경만하면 가능
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# %%
# 임베딩을 사용했던  RNN 모델과 완전동일 
# 모델구조 출력
model.summary()
# SimpleRNN클래스의 모델 파라미터 개수는 200개였는데 LSTM셀에는 작은열이 4개 있으므로 정확히 4배가 늘어 파라미터 개수가 800개가 됨

# %%
# 모델 컴파일후 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data =(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# %%
# 훈련손실과 검증 손실 그래프 출력
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
# 기본 순환층보다 LSTM이 과대적합을 억제하면서 훈련을 잘 수행한 것으로보임
# 하지만 경우에 따라서는 과대적합을 더 강하게 제어할 필요가 있음
# 드롭아웃 적용해볼까?

# %%
# 순환층에 드롭아웃 적용하기
# 완전 연결 신경망과 합성곱 신경망에서는 Dropout클래스를 사용해 드롭아웃을 적용했다
# 이를 통해 모델이 훈련세트에 너무 과대적합되는 것을 막았음
#  순환층은 자체적으로 드롭아웃 기능을 제공
# 전체적인 모델구조는 이전과 동일 LSTM클래스에 dropout 매개변수를 0,3으로 지정하여 30%의 입력을 드롭아웃
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# %%
# 이전과 동일하게 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data =(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
# 검증 손실이 약간 향상됨

# %%
# 훈련손실과 검증 손실 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

# 훈련손실과 검증 손실간의 차이가 좁혀 진것을 확인할수 있음
# 밀집층이나 합성곱 층처럼 순환층도 여러 개를 쌓지 않을 이유가없음

# %%
#2개의 층을 연결하기
# 순환층을 연결할 때는 한가지 주의할 점이있는데 은닉상태는 샘플의 마지막 타임스텝에 대한 은닉 상태만 다음 층으로 전달한다
# 하지만 순환층을 쌓게 되면 모든 순환층에 순차 데이터가 필요 따라서 앞쪽의 순환층이 모든 타임스텝에 대한 은닉 상태를 출력해야됨
# 오직 마지막 순환층만 마지막 타임스텝의 은닉 상태를 출력해야됨
# 케라스의 순환층에서 모든 타임스텝의 은닉 상태를 출력하려면 마지막을 제외한 다른 모든 순환층에서 return_sequences 매개변수를 True로 지정하면됨

model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))
# 2개의 LSTM층을 쌓았고 모두 드롭아웃 0,3으로 지정

# %%
# 결과 확인
model3.summary()

# %%
# 재훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model3.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data =(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
# 일반적으로 순환층을 쌓으면 성능이 높아짐
# 이 훈련에서는 그리 큰 효과를 내지못함 

# %%
# 손실 그래프 를 통해 과대적합이 잘 제어되었는지 확인
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
# %%

# GRU 구조
# GRU = Gated Recurrent Unit의 약자
# 해당 셀에는 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어있다. 
# 2개는 시그모이드 활성화 함수를 사용하고 하나는 tanh 활성화 함수를 사용
# GRU셀은 LSTM보다 가중치가 적기 때문에 계산량이 적지만 LSTM못지않은 좋은 성능을 낸다

# GRU 신경망 훈련
model4 =keras.Sequential()
model4.add(keras.layers.Embedding(500,16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))


# %%
model4.summary()
# GRU셀에는 3개의 작은셀이 있는데 입력가 은닉 상태에 곱하는 가중치와 절편이 있음
# 입력에 곱하는 가중치는 16 x 8 = 128이고 은닉 상태에 곱하는 가중치는 8 x 8 = 64이다
# 그리고 절편은 뉴런마다 하나씩이므로 8개 모두더하면 200개
# 이런 작은셀이 3개있으니 총 600개
# 결과는 623
# 텐서에서는 기본적으로 구현된 gru셀의 계산이 조금다름 그래서 결과가 다르게나옴
# 널리 통용되는 이론과 구현이 차이나는 경우가 종종있음

# %%
# GRU 신경망 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model4.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data =(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# %%
# 마찬가지로 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
# 드롭아웃을 사용하지 않았기 때문에 훈련 손실과 검증 손실 사이에 차이가 있지만 훈련과정이 잘 수렴되는걸 확인

#%%
# 가장 좋앗던 2개의 순환층을 쌓은 모델을 다시 로드하여 테스트 세트에 대한 성능을 확인

test_seq = pad_sequences(test_input, maxlen=100)
rnn_model = keras.models.load_model('best-2rnn-model.h5')
rnn_model.evaluate(test_seq, test_target)

# 이 모델은 드롭아웃을 적용하여 과대적합을 잘 억제했기 때문에 테스트 세트의 성능이 검증 세트와 크게 차이 나지않음
# 모델 학습간 사용한 라이브러리는 따로 정리하여 정리파일을 작성하는게 훨씬 유리하며 추후 다시 복습할땐 라이브러리마다의 장점에 대해 기술할 예정 
