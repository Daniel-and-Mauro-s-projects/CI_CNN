from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Preprocessing
file_path = './caltech101_silhouettes_28.mat'
data = loadmat(file_path)

# Plot an example
""" 
index = 10
image = data["X"][index].reshape((28, 28)).T
label = data["Y"][0][index]
plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale
plt.title(f"28x28 Image with label {data["classnames"][0][label-1][0]} with class number {label}")
plt.axis('off')
plt.show()
"""

images = np.array([data["X"][index].reshape((1, 28, 28)).T for index in range(len(data["X"]))])
labels = np.array(data["Y"][0])

# Creating the model

def create_cnn_model(nb_blocks, filters, activation_fn, input_shape, output_classes):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    for i in range(nb_blocks):
        model.add(layers.Conv2D(filters[i], kernel_size=(3, 3), activation=activation_fn, padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(output_classes, activation='softmax'))

    return model

def evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs, batch_size):
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, 
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_accuracy

def run_experiments(data, labels, configs, epochs=20, batch_size=32):
    results = []

    for config in configs:
        nb_blocks = config['nb_blocks']
        filters = config['filters']
        activation_fn = config['activation_fn']
        data_split = config['data_split']

        # Data Splitting
        x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=(1-data_split[0]))
        val_split = data_split[1] / (data_split[1] + data_split[2])
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(1-val_split))

        # Encode Labels
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)
        y_test = tf.keras.utils.to_categorical(y_test)

        # Run Model 3 Times
        accuracies = []
        for _ in range(3):
            model = create_cnn_model(nb_blocks, filters, activation_fn, data.shape[1:], labels.max() + 1)
            loss, accuracy = evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs, batch_size)
            print(f"Evaluated {nb_blocks}, {filters}, {activation_fn} with accuracy {accuracy}")
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        results.append({
            'config': config,
            'mean_accuracy': mean_accuracy
        })

    return pd.DataFrame(results)


nb_blocks_options = [1, 3]
filters_options = [
    [128],                    # with nb_blocks=1
    [32, 64, 128]             # with nb_blocks=3
]
activation_fn_options = ['sigmoid', 'relu']
data_split_options = [(0.8, 0.1, 0.1), (0.4, 0.2, 0.4), (0.1, 0.1, 0.8)]


configs = []
for nb_blocks, filters in zip(nb_blocks_options, filters_options):
    for activation_fn, data_split in itertools.product(activation_fn_options, data_split_options):
        configs.append({
            'nb_blocks': nb_blocks,
            'filters': filters,
            'activation_fn': activation_fn,
            'data_split': data_split
        })

results_df = run_experiments(images, labels, configs)
print(results_df)