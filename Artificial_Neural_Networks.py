# 인공 신경망 
# 인공 신경망이란 무엇인가에 대한 공식적인 정의는 없다 
# 만약 통계학적 모델의 집합이 다음과 같은 특징들을 가진다면 해당 집합을 신경(neural)이라고부른다
# 1. 조정이 가능한 가중치들의 집합 즉 학습 알고리즘에 의해 조정이 가능한 숫자로 표현된 매개변수로 구성
# 2. 입력의 비선형 함수를 유추가능
#%%
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
#  keras.datasets.fashion_mnist 모듈 아래 load_data함수가 훈련데이터와 테스트데이터를 나누어 반환
# %%
# 전달 받은 데이터의 크기 확인
print(train_input.shape, train_target.shape)
# 총 6만개의 이미지로 이루어짐 각 이미지는 28 x 28 크기 1차원배열
# %%
print(test_input.shape, test_target.shape)
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
# %%
print([train_target[i] for i in range(10)])
# 패션 MNIST의 타깃은 0~9까지의 숫자 레이블로 구성된다 . 
# %%
import numpy as np
print(np.unique(train_target, return_counts=True))
# %%
# 로지스틱 회귀로 패션 아이템 분류하기
# 훈련 샘플이 6만개나 되기 떄문에 전체 데이터를 한거번에 사용하여 모델을 훈련하는것보다 샘플을 하나씩 꺼내서 모델을 훈련하는 방법이 더 효율적이다
# 넘파이 배열의 nbytes 속성에 실제 해당 배열이 차지하는 바이트 용량이 저장되어있다 

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
# %%
print(train_scaled.shape)
# %%
# 교차검증으로 성능을 확인
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
# 반복횟수 5번으로 지정
# %%
# 가장 기본적인 인공 신경망은 확률적 경사 하강법을 사용하는 로지스틱 회귀와 같다
import tensorflow as tf
from tensorflow import keras

# 인공신경망으로 모델만들기
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)
# %%
print(val_scaled.shape, val_target.shape)
# %%
# 케라스의 레이어 패키지 안에는 다양한 층이 준비되어있는데 가장 기본이 되는 층은 밀집층이다
# 픽셀과 뉴런이 연결되어있는 층을 양쪽이 뉴런이 모두 연결하고 있기 떄문에 완전 연결층 이라고 부른다

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784, ))
# 10 = 뉴런개수 , activation = 뉴런의 출력에 적용할 함수 , input_shape= 입력의크기

# %%
model = keras.Sequential(dense)
# 절편의 경우는 아예 선도 그리지 않는 경우가 많다. 하지만 절편이 뉴런마다 더해진다는 것을 꼭 기억해야됨
# 소프트맥스와 같이 뉴런의 선형 방정식 계산 결과에 적용되는 함수는 활성화 함수라고 부른다

# %%
# 인공 신경망으로 패션 아이템 분류하기
model.compile(loss='sparse_categorical_crossentropy', 
metrics='accuracy')
# 해당 모델 컴파일 
# 훈련시킨 모델을 컴파일을 해야 test가 가능
# 다중 분류에서는 크로스 엔트로피 손실 함수 사용 
# 케라스에서는 이 두 손실 함수를 가각 binary_crossentropy categorical_crossentropy 로 나누어 부른다
# 이진분류 : loss = binary_crossentropy 
# 다중 분류: loss= categorical_crossentropy

#%%
print(train_target[:10])

#%%
model.fit(train_scaled, train_target, epochs=5)
# 케라스는 친절하게 에포크마다 걸린 시간과 손실, 정확도를 출력해줌
# 케라스에서 모델의 성능을 평가하는 메서드는 evaluate()메서드
# %%
model.evaluate(val_scaled, val_target)
# 검증세트의 점수는 훈련 세트 점수보다 조금 낮은것이 일반적임

# 원-핫 인코딩 = 정숫값을 배열에서 해당 정수 위치의 원소만 1이고 나머지는 모두 0으로 변환
# 이런 변환이 필요한 이유는 다중 분류에서 출력층에서 만든 확률과 크로스 엔트로피 손실을 계산하기위해
# %%
