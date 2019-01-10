# %% Cell 1
import tensorflow as tf

#Imports the data for training
mnist = tf.keras.datasets.mnist #28 * 28 images of hand written digits 0-9

#unpacks the data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scales the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#builds the model, (network)

#creates input layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#creates hidden layers
#activation function is like the sigmoid, 128 is setting the size
model.add(tf.keras.layers.Dense(256, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))

#creates output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 2)

# %% Cell 2
#measure loss
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss Value: ", val_loss * 100, "| Accuracy Value", val_acc * 100)

# %%Cell 3
print("why are you doing this?")

# %%Cell 4
"""
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])
"""