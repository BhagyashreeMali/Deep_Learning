import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Preprocess
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Display sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(names[np.argmax(y_train[i])])
    plt.axis("off")
plt.show()

# CNN model
model = models.Sequential([
    layers.Input((32,32,3)),
    layers.Conv2D(32,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10,activation="softmax")
])

model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Train
h = model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.1)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,verbose=0)
print("Test loss:",loss)
print("Test accuracy:",acc*100,"%")

# Predict 10 images
p = np.argmax(model.predict(x_test[:10],verbose=0),axis=1)
actual = np.argmax(y_test[:10],axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"P:{names[p[i]]}\nA:{names[actual[i]]}",
              color="green" if p[i]==actual[i] else "red")
    plt.axis("off")
plt.show()

# Accuracy graph
plt.plot(h.history["accuracy"],label="Training")
plt.plot(h.history["val_accuracy"],label="Validation")
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Loss graph
plt.plot(h.history["loss"],label="Training")
plt.plot(h.history["val_loss"],label="Validation")
plt.title("Loss")
plt.legend()
plt.grid()
plt.show()import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Preprocess
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Display sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(names[np.argmax(y_train[i])])
    plt.axis("off")
plt.show()

# CNN model
model = models.Sequential([
    layers.Input((32,32,3)),
    layers.Conv2D(32,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10,activation="softmax")
])

model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Train
h = model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.1)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,verbose=0)
print("Test loss:",loss)
print("Test accuracy:",acc*100,"%")

# Predict 10 images
p = np.argmax(model.predict(x_test[:10],verbose=0),axis=1)
actual = np.argmax(y_test[:10],axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"P:{names[p[i]]}\nA:{names[actual[i]]}",
              color="green" if p[i]==actual[i] else "red")
    plt.axis("off")
plt.show()

# Accuracy graph
plt.plot(h.history["accuracy"],label="Training")
plt.plot(h.history["val_accuracy"],label="Validation")
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Loss graph
plt.plot(h.history["loss"],label="Training")
plt.plot(h.history["val_loss"],label="Validation")
plt.title("Loss")
plt.legend()
plt.grid()
plt.show()import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Preprocess
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Display sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(names[np.argmax(y_train[i])])
    plt.axis("off")
plt.show()

# CNN model
model = models.Sequential([
    layers.Input((32,32,3)),
    layers.Conv2D(32,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10,activation="softmax")
])

model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Train
h = model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.1)

# Evaluate
loss,acc = model.evaluate(x_test,y_test,verbose=0)
print("Test loss:",loss)
print("Test accuracy:",acc*100,"%")

# Predict 10 images
p = np.argmax(model.predict(x_test[:10],verbose=0),axis=1)
actual = np.argmax(y_test[:10],axis=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"P:{names[p[i]]}\nA:{names[actual[i]]}",
              color="green" if p[i]==actual[i] else "red")
    plt.axis("off")
plt.show()

# Accuracy graph
plt.plot(h.history["accuracy"],label="Training")
plt.plot(h.history["val_accuracy"],label="Validation")
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Loss graph
plt.plot(h.history["loss"],label="Training")
plt.plot(h.history["val_loss"],label="Validation")
plt.title("Loss")
plt.legend()
plt.grid()
plt.show()
