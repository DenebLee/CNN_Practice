# 딥러닝에서는 모델의 구조를 직접 만든다는 느낌이 훨씬 강하다
# 케라스의 fit() 메서드는 History 클래스 객체를 반환한다
# 손실 곡선
#%%
from threading import active_count
from google.protobuf.descriptor import EnumValueDescriptor
from numpy.lib.histograms import histogram
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target =train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
#%%
# 모델 생성
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
model = model_fn()
model.summary()
# %%
# 모델을 훈련하지만 fit()메서드의 결과를 history 변수에 담기
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

# %%
# verbpse = 0 이란
# verbose 매개변수는 훈련 과정 출력을 조절 기본값 1로 이전 절에서처럼 에포크마다 진행 막대와 함께 손실등의 지표가 출력된다 
# 2로 바꾸면 진행 막대를 빼고 출력
print(history.history.keys())
# history 딕셔너리엔 손실과 정확도가 포함
# 케라스는 기본적으로 에포크마다 손실을 계산

# %%
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# 손실 그래프
# %%
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
# 정확도 그래프
# 에포크마다 손실이 감소하고 정확도가 향상
# %%
# 에포크가 5번뿐이기에 늘리면 정확도는 늘고 손실이 줄어들것이라는 가정
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# 에포크 횟수가 늘어나면 정확도는 늘게되며 손실이 줄어든다 확인! 
# %%
# 인공 신경망 모델이 최적화하는 대상은 정확도가 아니라 손실 함수다
# 이따금 손실 감소에 비례하여 정확도가 높아지지 않는경우가 있다
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
# %%
print(history.history.keys())

#검증 세트에 대한 손실은 val_loss에 들어있고 정확도는 val_accuracy에 있다

# %%
# 과대/과소적합 문제를 조사하기 위해 훈련 손실과 검증 손실을 한 그래프에 그려서 비교
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 초기에 검증 손실이 감소하다가 다섯 번쨰 에포크 만에 다시 상승
# 훈련 손실은 꾸준히 감소하기 떄문에 전형적인 과대적합 모델이 만들어짐
# %%
# 옵티마이저 하이퍼파라미터를 조정하여 과대적합을 완화시킬수 있는지 확인
# adam은 적응적 학습률을 사용하기 떄문에 에포크가 진행되면서 학습률의 크기를 조정가능
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %%
# 드롭아웃 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서 과대적합을 막는다
# 이전 층의 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과대하게 의존하는것을 줄일 수 있고 모든 입력에 대해 주의를 기울여야 함
# 일부 뉴런의 출력이 없을 수 있다는 것을 감안하면 이 신경망은 더 안정적인 예측을 만들수 있을것

# 30% 정도 드롭아웃한 모델
model = model_fn(keras.layers.Dropout(0.3))
model.summary()
# 결과에서 볼 수 있듯이 은닉층 뒤에 추가된 드롭아웃층은 훈련되는 모델 파라미터가 없음
# 일부 뉴런의 출력을 0으로 만들지만 전체 출력배열의 크기를 바꾸지는 않음
# 훈련이 끝난 뒤에 평가나 예측을 수행할 떄는 드롭아웃을 적용하지 말아야함
# 훈련된 모든 뉴런을 사용해야 올바른 예측을 수행할수 있음
# 텐서플로와 케라스는 모델을 예측이나 평가에 사용할때는 자동으로 드롭아웃을 적용 x
# 훈련 손실과 검증 손실의 그래프
#%%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 과대적합이 확실히 줄었음  (열 번째 에포크 정도에서 검증 손실의 감소가 멈추지만 크게 상승하지않고 어느정도 유지되는중)
# 이모델은 20번의 에포크 동안 훈련을 했기 떄문에 결국 다소 과대적합되어있다

# %%
# 에포크 10회로 제한후 재훈련
# 재훈련후 저장하기
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data=(val_scaled, val_target))

# %%
model.save_weights('model-weights.h5')
#해당 모델 저장 
# HDF5 포맷으로 저장
model.save('model-whole.h5')
# 두가지 파일 생성확인필요 -----> 생성완료
# %%
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

# 훈련하지 않은 새로운 모델을 만들고 이전에 저장했던 모델 파라미터를 적재
# %%
import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))
# %%
# 모델 전체를 파일에서 읽은 다음 검증 세트의 정확도 출력해보기
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)

# 같은 모델을 저장하고 다시 불러들였기 떄문에 위와 동일한 정확도를 얻음
# 과정을 돌이켜 보면 20번의 에포크 동안 모델훈련하여 검증 점수가 상승하는 지점을 확인
# 그다음 과대적합 되지않는 에포크만큼 다시 훈련
# 모델을 두번 훈련하지않고 한번에 끝낼수 있는방법 = 케라스의 콜백

# %%
# 콜백 : 훈련과정 중간에 어떤 작업을 수행할 수 있게하는 객체
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])

# %%
# 해당 모델을 load_model()함수로 다시 읽어서 예측 수행
model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)

# 검증 점수가 상승하기 시작하면 그 이후에는 과대적합이 더 커기지 때문에 훈련을 계속할 필요가 없음
# 이때 훈련을 중지하면 컴퓨터 자원과 시간을 아낄수 있음
# 이런 방법을 조기 종료라고하며 딥러닝 분야에서 널리 사용

# %%
# EarlyStopping 콜백을 ModelCheckpoint 콜백과 함께 사용하면 가장 낮은 검증손실의 모델을 파일에 저장하고 검증 손실이 다시 상승할 때 훈련을 중지가능
# 두 콜백 사용
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# %%
print(early_stopping_cb.stopped_epoch)
# 14번쨰 에포크 진행 과정에서 훈련이 중지됨 patience를 2로 지정했으므로 최상의 모델은 현재 출력된 숫자번째 에포크일것

# %%
# 훈련 손실과 검증 손실 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# %%
# 조기종료로 얻은 모델을 사용해 검증 세트에 대한 성능 확인
model.evaluate(val_scaled, val_target)
# %%
