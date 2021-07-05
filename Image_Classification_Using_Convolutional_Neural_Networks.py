# 패션 mnist 데이터 불러오기
# 합성곱 층은 여러 개의 필터를 사용해 이미지에서 특징을 학습한다 
# 쉽게 말해 마치 입력 데이터에 도장을 찍어서 유용한 특성만 드러나게 하는 것으로 비유할 수 있다. 
# 인공 신경망은 처음에 가중치 w1 ~ w10과 절편 b를 랜덤하게 초기화한 에포크를 반복 하면서 경사 하강법 알고리즘을 사용하여 손실이 낮아지도록 최적의 가중치와 절편을 찾아간다
# 예를 들어 밀집층에 뉴런이 3개면 출력은 3개가도니다 입력갯수 상관x
# 합성곱은 밀집층의 계산과 조금 다르다. 입력 데이터 전체에 가중치를 적용하는 것이 아니라 임무에 가중치를 곱함
# 합성곱 신경망에서는 완전 연결 신경망과 달리 뉴런을 필터라고 부름 혹은 커널

# 합성곱 계산을 통해 얻은 출력을 특별히 특성맵map이라고부름

# 필터 , 커널, 특성 맵 에 대한 용어 정리는 필수로 알아둬야됨

# 패딩 = 합성곱 층의 입력 주위에 추가한 0으로 채워진 픽셀 패딩을 사용하지 않는 것을 벨리드 패딩
# 스트라이드 = 합성곱 층에서 필터가 입력 위를 이동하는 크기 일반적으로 1px사용
# 풀링 = 가중치가 업속 특성 맵의 가로세로 크기를 줄이는 역활을 수행

# 합성곱 신경망을 사용한 이미지 분류 
# 패션 MNIST 데이터 불러오기
#%%
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 정말 중요한거지만 입력 이미지는 항상 depth가 있어야됨
# %%
# 합성곱 신경망 만들기 
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))


# %%
model.add(keras.layers.MaxPooling2D(2))
# 이미지가 (28, 28) 크기에 세임 패딩을 적용했기에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 입력과 동일

# %%
# 첫번째 합성곱 -풀링 층 다음에 두번째 합성곱 -풀링 층 추가
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# %%
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
# 은닉층과 출력층 사이에 드롭아웃 넣음
# %%
# 모델 구조 출력
model.summary()
# %%
# 앞에서 만든 model객체를 넣어 호출
keras.utils.plot_model(model)

# pip install pydot
# pip install graphviz 
# vscode에선 안되는데 코랩에선 가능
# %%
keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)
# %%
# 모델 컴파일과 훈련
# 케라스 api의 장점은 딥러닝 모델의 종류나 구성 방식에 상관없이 컴파일과 훈련 과정이 같다는 점
# Adam 옵티마이저를 사용하고 ModelCheckpoint 콜백과 EarlyStopping 콜백을 함께 사용해 조기 종료 기법을 구현
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20,
validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# 회사 컴퓨터론 도저히 안돌아감
# %%

# 손실 그래프
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# %%
# 세트에 대한 성능 평가
model.evaluate(val_scaled, val_target)
#%%
# 첫번째 샘플 이미지 확인
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

#%%
preds =model.predict(val_scaled[0:1])
print(preds)

# 출력 결과를 보면 아홉 번째 값이 1이고 다른 값은 거의 0에 가까움

#%%
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()
# 다른 클래스의 값은 사실상 모두 0

#%%
# 파이썬에서 레이블을 다루기 위해 리스트로 저장
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

#%%
import numpy as np
print(classes[np.argmax(preds)])

#%%
# 훈련 세트와 검증 세트에서 했던 것처럼 픽셀값의 범위를 0~1 사이로 바꾸고 이미지 크기를 (28, 28)에서 (28,28,1)로 변경
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)

# 테스트 세트에서의 점수는 검증 세트보다 조금 더 작음
# %%
