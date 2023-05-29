from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np


x = np.fromfile('x.csv', sep=',')
y = np.fromfile('y.csv', sep=',')

x = np.reshape(x, (100_000, 70))

print(x[0])
print(y)


model = keras.Sequential([
    keras.layers.Dense(70, input_shape=(70,)),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

history = model.fit(x, y, epochs=25, validation_split=0.1)

model.save('model')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
