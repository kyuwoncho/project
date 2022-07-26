import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#local에 있는 커스텀 데이터 디렉토리
data_dir = 'C:/Users/mtr09/PycharmProjects/drink/images/data'

#이미지를 가져올 크기와 배치 사이즈
img_height = 180
img_width = 180
batch_size = 10

#train 이미지 디렉토리로 부터 가져와서 batch, image_size를 통해 처리
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'images/data/',
    labels = 'inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split=0.2,
    subset="training",

)
#검증용 이미지도 위와 마찬가지로 적용
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'images/data/',
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",

)
#class_names에는 폴더명을 그대로 label로 사용한 'coke'와 'pepsi'가 있음.
class_names = train_ds.class_names
print(class_names)


#픽셀정보가 0~255사이의 값인데 학습을 위해서 0~1사이의 값으로 정규화하는과정.
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#정규화의 결과를 예시로 보여주는 코드.
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

#분류할 클래스 개수 pepsi와 coke 2개임.
num_classes = 2

#model 생성.
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
#모델 컴파일 optimizer와 loss함수와 accuracy표시를 정해준다.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#모델의 weight 수를 확인할수있다
model.summary()
epochs = 10
#fit 함수로 학습.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#여기는 실제로 학습된 모델을 통해서 예측하는 과정. 결과를 얻고싶은 이미지를 입력으로 주면 결과로 분류해줌.
image_path = 'C:/Users/mtr09/Desktop/t_pepsi.jpg'
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

#predict함수가 결과예측해줌.
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

