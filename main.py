import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3
pool_size = 2

accuracy = {}
# Tests for accuracy
for i in range(1, 16):
    print("Testing with epochs: " + str(i))
    # Build the model.
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ])
    # Compile the model.
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model.
    history = model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=i,
        validation_data=(test_images, to_categorical(test_labels)),
    )

    # Save the model to disk.
    model.save_weights('mnist_cnn.h5')

    # Load the model from disk later using:
    # model.load_weights('mnist_cnn.h5')

    # Predictions initialization
    predictions = model.predict(test_images[:30])

    # Print our model's predictions.
    print(np.argmax(predictions, axis=1))

    # Check our predictions against the ground truths.
    print(test_labels[:30])
    accuracy[i] = history.history['accuracy']

print(accuracy)
