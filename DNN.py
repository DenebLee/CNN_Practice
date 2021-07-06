# 심층 신경망 DNN
# 입력층과 출력층 사이에 여러개의 은닉층들로 이뤄진 인공신경망이다. 
# 일반적인 인공신경망과 마찬가지로 복잡한 비선형관게들을 모델링 할수 있다
# 2개의 층
#%%
from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import adagrad
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# 케라스 api를 사용해서 데이터셋 불러오기
# %%
from sklearn.model_selection import train_test_split
train_scaled =train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target =train_test_split(train_scaled, train_target, test_size= 0.2, random_state=42)

# %%
# 인공신경망 모델에 층을 2개 추가하기
# 은닉층 에는 주황색 원으로 활성화 함수가 표시되어 있는데 이는 신경망 층의 선형 방정식의 계산 값에 적용하는 함수
# 분류 문제는 클래스에 대한 확률을 출력하기 위해 활성화 함수를 사용

# 인공신경망을 그림으로 표현할때 활성화 함수를 표현안하는 경우가 있는데 모든 신경망 은닉층에는 활성화 함수가 있다 

dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
# dense1이 은닉층이고 100개의 뉴런을 가진 밀집층
# 몇개의 뉴런을 두어야 할지 판단하기 위해서는 상당한 경험이 필요
# 한가지 제약 사항이 있다면 적어도 출력층의 뉴런보다는 많게 만들어야 한다

# dense2는 출력층
# 10개의 클래스를 분류하므로 10개의 뉴런을 두고 활성화 함수는 소프트맥스함수 사용
dense2 = keras.layers.Dense(10, activation='softmax')
# %%
# 심층 신경망 만들기 
# dense1 dense2 객체를 sequential 클래스에 추가하여 심층 신경망 제작
model = keras.Sequential([dense1, dense2])
# %%
# 케라스는 모델의 summary()메서드를 통해 층에 대한 유용한 정보를 얻을수 있다 
model.summary()
# %%
# 층을 추가하는 다른방법 
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,) , name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')], name=' 패션 MNIST 모델')

#이렇게 작업하면 추가되는 층을 한눈에쉽게 알아보는 장점이있다 
# %%
model.summary()

# 2개의 Dense층이 이전과 동일하게 추가되었고 파라미터 개수도 같다
# 바뀐 것은 모델 이름과 층 이름
# 이 방법이 편리하지만 아주 많은 층을 추가하려면 Sequential 클래스 생성자가 매우 길어진다
# 또 조건에 따라 층을 추가할수 없다
# 층을 추가할때 가장 널리 사용하는 방법은 모델의 add()메서드

# %%
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid',input_shape=(784,)))
model.add(keras.layers.Dense(100, activation='softmax'))
# %%
model.summary()

# %%
#모델 훈련하기
model.compile(loss='sparse_categorical_crossentropy' , metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# %%
# 렐루 함수
# 초창기 인공 신경망의 은닉층에 많이 사용된 활성화 함수는 시그모이드 함수
# 함수의 오른쪽과 왼쪽 끝으로 갈수록 그래프가 누워있기 때문에 올바른 출력을 만드는데 신속하게 대응하지 못한다
# 특히 층이 많은 심층 신경망일수록 그 효과가 누적되어 학습을 더어렵게 하는데 이를 개선한 함수가 렐루 함수

# 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 통과시키고 음수일 경우에는 0으로만든다

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
# %%
model.summary()
# %%
# 훈련 데이터를 다시 준비해서 모델을 훈련 
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size= 0.2, random_state=42)

# %%
# 모델을 컴파일 하고 훈련하는것은 이전과 동일
model.compile(loss='sparse_categorical_crossentropy' , metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# %%
# 검증 세트에서의 성능 확인
model.evaluate(val_scaled, val_target)
# %%
# 옵티마이저 
# 머신러닝 학습 프로세스에서 실제로 파라미터를 갱신시키는 부분을 의미 
# 케라스는 기본적으로 미니배치 경사 하강법을 사용하며 미니배치 개수는 32개이다

model.compile(optimizer='sgd', loss='sparse_categorucal_crossentropy', metrics='accuaracy')

# 이 옵티마이저는 tensorflow.keras.optimizers 패키지 아래 SGD클래스로 구현되어있다. 
# 'sgd' 문자열은 이 클래스의 기본 설정 매개변수로 생성한 객체와 동일

# %%
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

# 원래 sgd= keras.optimizers.SGD()처럼 SGD클래스 객체를 만들어 사용해야 하는데 번거로움을 피하고자 'sgd'라고 지정하면 자동으로 SGD객체를 만들어줌
# %%
sgd = keras.optimizers.SGD(learning_rate=0.1)

#기본 경사 하강법 옵티마이저는 모두 SGD클래스에서 제공 
# 0보다 큰값으로 지정하면 마치 이전의 그레이디언트를 가속도처럼 사용하는 모멘텀 최적화를 사용
# 다음처럼 SGD클래스의 nesterov 매개변수를 기본값 False에서 True로 바꾸면 네스테로프 모멘텀 최적화 또는 네스테로프 가속 경사를 사용

# %%
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
# %%
# 모델이 최적점에 가까이 갈수록 학습률을 낮출수 있다. 이렇게하면 안정적으로 최적점에 술며할 가능성이 높다 
# 이러한 학습률을 적응적 학습률이라고한다
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')
# %%
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')

# %%
# 모델 다시 생성
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
# optimizer를 'adam'으로 설정하고 5번의 에포크동안 훈련
# %%
# 성능 테스트
model.evaluate(val_scaled, val_target)
# %%
# 정리
# 옵티마이저는 신경망의 가중치나 절편을 학습하기 위한 알고리즘 또는 방법을 말함