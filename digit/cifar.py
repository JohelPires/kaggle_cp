import numpy as np
import tensorflow as tf

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



import matplotlib.pyplot as plt

plt.subplot(121)
plt.imshow(x_train[0])
plt.subplot(122)
plt.imshow(x_train[1])
plt.show()



model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print(loss)
print(acc)