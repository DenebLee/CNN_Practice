# 가중치 시각화 
# 합성곱은 여러 개의 필터를 사용해 이미지에서 특징을 학습
# 각 필터는 커널이라고 부르는 가중치와 절편을 가지고있는데 일반적으로 절편은 시각적으로 의미가 있지않는다
# 가중치는 입력 이미지의 2차원 영역에 적용되어 어떤 특징을 크게 두드러지게 표현하는 역활을 한다

# 체크포인트 파일 읽어들이기
#%%
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')
# 케라스 모델에 추가한 층은 layers 속성에 저장되어 있다. 이속성은 파이썬 리스트

# %%
# model.layers 출력
model.layers
# model.layers 리스트에 추가했던 Conv2D, MaxPooling2D 층이 번갈아 2번 연속 등장함

# %%
# 첫번째 합성곱 층의 가중치를 조사
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)

# 합성곱 층에 전달되는 입력의 깊이가 1이므로 실제 커널의 크기는 (3,3,1)
# weights 속성은 텐서플로의 다차원 배열인 Tensor 클래스의 배열
#%%
conv_weights =conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# 이 가중치의 평균값은 0에 가깝고 표준편차는 0.27정도 
# %%
# 가중치가 어떤 분포를 가졌는지 직관적으로 이해하기 쉽도록 히스토그램 그려보기
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 중요*** 맷플롯립의 hist()함수에는 히스토그램을 그리기 위해 1차원 배열로 전달해야함
# 이를위해 reshape 메서드로 conv_weights 배열을 1개의 열이 있는 배열로 변환

# %%
# 32개의 그래프 영역을 만들고 순서대로 커널을 출력

fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i  in range(2):
    for j in range(16):
        axs[ i, j].imshow(conv_weights[: , : , 0, i*16+ j ], vmin=-0.5, vmax=0.5)
        axs[ i , j].axis('off')
plt.show()
# 32개의 가중치를 저장 이 배열의 마지막 차원을 순회하면서 0부터 i *16 + j번째 까지 가중치의 값을 차례대로 출력
# 여기에서는 i가 행 인덱스 j 가 열 인덱스 각각 0~1, 0~15까지의 범위를 가짐
# 그래프를 보면 가중치의 값이 무작위로 나열된 값이 아닌 어떤 패턴을 볼수있음
# 예를들어 첫번쨰 줄의 맨 왼쪽 자우치는 오른쪽 3픽셀의 값이 높다

# %%
# 훈련하지 않은 반 합성곱 신경망 제작
# Sequential 클래스로 모델을 만들고 Conv2D 층을 하나 추가
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))

#%%
# 모델의 첫 번째 층( Conv2D)의 가중치를 no_training_conv 변수에 저장
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)
# %%

no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# %%
# 평균은 이전과 동일하게 0에 가깝지만 표준편차는 이전과 달리 매우적음
# 히스토그램으로 표현
plt.hist(no_training_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 이전과 확실히 다름
# 대부분의 가충치가 -0.15~ 0.15 사이에 있고 비교적 고른 분포를 보임
# 이런 이유는 텐서플로가 신경망의 가중치를 처음 초기화할 때 균등 분포에서 랜덤하게 값을 선택하기 떄문

# %%
# 이 가중치의 값을 맷플롯립의 inshow() 함수를 사용해 그램으로 출력
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i  in range(2):
    for j in range(16):
        axs[ i, j].imshow(conv_weights[: , : , 0, i * 16 + j ], vmin=-0.5, vmax=0.5)
        axs[ i , j].axis('off')
plt.show()
# 전체적으로 가중치가 밋밋하게 초기화되었음
# 합성곱 신경망이 패션 MNIST 데이터셋의 분류 정확도를 높이기 위해 유용한 패턴을 학습했다는 사실을 눈치챌수 있음

# %%
# 함수형 API
# 지금까지 신경망 모델을 만들 때 케라스 Sequential클래스를 사용
# 이 클래스는 층을 차례대로 쌓은 모델을 만듬
# 딥러닝에서는 좀 더 복잡한 모델이 많이 있는데 입력이 2개일 수도 있고 출력이 2개일 수도있음 
# 이럴경우 해당 함수를 사용하기 어려움 대신 함수형 api를 사용
# 함수형 api는 케라스의 Model 클래스를 사용하여 모델을 만든다. 
# 앞서 만들었던 Dense 층 2개로 이루어진 완전 연결 신경망을 함수형 api로 구현해보기

# dense1 = keras.layers.Dense(100, activation='sigmoid')
# dense2 = keras.layers.Dense(10, activation='softmax')

# 앞서 봤던것과 거의 동일
# 이객체를 Sequential클래스 객체의 add()메서드에 전달할 수 있다.
# 하지만 다음과 같이 함수처럼 호출할 수 있음

# hidden = dense1(inputs)

# 파이썬의 모든 객체는 호출 가능 
# 케라스의 층은 객체를 함수처럼 호출했을때 적절히 동작할 수 있도록 미리 준비
# 두번째 층을 호출함으로써 첫번쨰 층의 출력을 입력으로 사용

# outputs = dense2(hidden)

# 그다음 inputs와  outputs을 model클래스로 연결해주기

# model =keras.Model(inputs,outputs)

# 입력에서 출력까지 층을 호출한 결과를 계속 이어주고  Model 클래스에 입력과 최종 출력을 지정
# Sequential 클래스는 InputLayer 클래스를 자동으로 추가하고 호출해 주지만 Model 클래스에서는 우리가 수동으로 만들어서 호출해야됨 
# Sequential 클래스는 InputLayer 클래스를 자동으로 추가하고 호출해 주지만 Model 클래스에서는 우리가 수동으로 만들어서 호출해야됨
# 바로 Input 이 Inputtlayer 클래스의 출력값이 되어야함

# 다행이 케라스는 InputLayer 클래스 객체를 쉽게 다룰 수 있도록 Input() 함수를 별도로 제공
inputs= keras.Input(shape=(784,))

#%% 
# model.input으로 모델의 입력을 간단히 얻을 수 있음
print(model.input)
# %%
conv_acti = keras.Model(model.input, model.layers[0].output)
# %%
# 특성 맵 시각화
# 케라스로 데이터셋을 읽은후 훈련 세트에 있는 첫 번쨰 샘플 그리기
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()


# %%
inputs = train_input[ 0 : 1 ].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)

# %%
# conv_acti.predict() 메서드가 출력한 feature_maps의 크기 확인
print(feature_maps.shape)
# 세임 패딩과 32개의 필터를 사용한 합성곱 층의 출력이므로 (28, 28, 32)이다.

# %%
# imshow() 함수로 이 특정맵을 그려보기
fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i  in range(4):
    for j in range(8):
        axs[ i, j].imshow(feature_maps[0, : , : ,  i * 8 + j ])
        axs[ i , j].axis('off')
plt.show()
# 이 특성 맵은 32개의 필터로 인해 입력 이미지에서 강하게 활성화된 부분을 보여줌
# 그림에서 첫 번쨰 필터는 오른쪽에 있는 수직선을 감지 
# 세 번째 필터는 전체적으로 밝은색이므로 전면이 모두 칠해진 영역을 감지
# 이와 반대로 마지막 필터는 전체적으로 낮은 음수 ㄱ밧
# 필터와 큰 양수가 곱해지면 더 큰 음수가 되고 배경처럼 0에 가까운 값과 곱해지면 작은 음수가 될것
# 즉 부츠의 배경이 상대적으로 크게 활성화 될수 있음

# %%
# model 객체의 입력과 두 번째 합성곱 층인 model.layers[2]의 출력을 연결한 conv2_ucti모델생성

conv2_acti = keras.Model(model.input, model.layers[2].output)

# %%
# 그다음 첫 번쨰 샘플을 conv2_acti모델의 predict() 메서드에 전달
inputs = train_input[ 0 : 1 ].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv2_acti.predict(inputs)
# %%
print(feature_maps.shape)

# %%
# 64개의 특성 맵을 8개씩 나누어 imshow()함수로 그려보기
fig, axs = plt.subplots(8, 8, figsize=(12, 12))
for i  in range(8):
    for j in range(8):
        axs[ i, j].imshow(feature_maps[0, : , : , i * 8 + j ])
        axs[ i , j].axis('off')
plt.show()
# 시각적으로 이해하기 어렵게 나옴
# 두 번째 합성곱 층의 필터 크기는 (3, 3, 32 )이다 .두 번째 합성곱 층의 첫 번째 필터가 앞서 출력한 32개의 특성 맵과 곱해져 두 번째 합성곱 층의 첫 번째 특성맵이 됨
# 이렇게 계산된 출력은 (14, 14, 32 ) 특성 맵에서 어떤 부위를 감지하는지 직관적으로 이해하기 어려움

# 이러한 현상은 합성곱 층을 많이 쌓을수록 심해짐
# 이를 바꾸어 생각해보면 합성곱 신경망의 앞부분에 있는 합성곱 층은 이미지의 시각적인 정보를 감지하고 뒤쪽에 있는 합성곱 층은 앞쪽에서 감지한 시각적 정보를 바탕으로 추상적인 정보를 학습한다고 볼수 있음