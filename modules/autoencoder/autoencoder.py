import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class Autoencoder:
    def __init__(self, neurons_per_layer, pretrain, pretrain_epochs, finetune_epochs, finetune_batch_size):
        """
        A TensorFlow 2.x implementation of a deep autoencoder.

        :param neurons_per_layer: List of integers specifying the number of neurons in each layer.
        :param pretrain: Boolean indicating whether to perform pretraining.
        :param pretrain_epochs: Number of epochs for pretraining.
        :param finetune_epochs: Number of epochs for fine-tuning.
        :param finetune_batch_size: Batch size for fine-tuning.
        """
        self.neurons_per_layer = neurons_per_layer
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.finetune_batch_size = finetune_batch_size
        self.autoencoder = self._build_autoencoder()

    def _build_autoencoder(self):
        """
        Builds the autoencoder model.
        """
        input_dim = self.neurons_per_layer[0]
        encoding_dim = self.neurons_per_layer[-1]

        # Encoder
        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = input_layer
        for neurons in self.neurons_per_layer[1:-1]:
            encoded = Dense(neurons, activation='sigmoid')(encoded)
        encoded_output = Dense(encoding_dim, activation='sigmoid')(encoded)

        # Decoder
        decoded = encoded_output
        for neurons in reversed(self.neurons_per_layer[1:-1]):
            decoded = Dense(neurons, activation='sigmoid')(decoded)
        decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

        # Autoencoder model
        autoencoder = Model(input_layer, decoded_output)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def train(self, data):
        """
        Trains the autoencoder on the given data.

        :param data: Input data to train the autoencoder.
        """
        # Reshape the data to 1D
        data = np.array(data, dtype=np.float32)  # Ensure data is a NumPy array
        data = data.reshape((data.shape[0], -1))  # Flatten each image into a 1D array

        # Normalize the data
        max_vals = np.max(np.abs(data), axis=1, keepdims=True)  # Compute max values for each sample
        data = np.where(max_vals != 0, data / max_vals, data)  # Normalize only if max_vals != 0
        np.random.shuffle(data)

        if self.pretrain:
            # Pretrain each layer individually
            for i in range(len(self.neurons_per_layer) - 1):
                print(f"Pretraining layer {i + 1}")
                layer_autoencoder = self._build_single_layer_autoencoder(i)
                layer_autoencoder.fit(data, data, epochs=self.pretrain_epochs, batch_size=self.finetune_batch_size, verbose=1)
                data = layer_autoencoder.predict(data)

        # Fine-tune the full autoencoder
        self.autoencoder.fit(data, data, epochs=self.finetune_epochs, batch_size=self.finetune_batch_size, verbose=1)

    def _build_single_layer_autoencoder(self, layer_index):
        """
        Builds a single-layer autoencoder for pretraining.

        :param layer_index: Index of the layer to pretrain.
        """
        input_dim = self.neurons_per_layer[layer_index]
        encoding_dim = self.neurons_per_layer[layer_index + 1]

        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='sigmoid')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def generate_encodings(self, data, labels, save_to_path):
        """
        Generates encodings for the input data and saves them to the specified path.

        :param data: Input data to encode.
        :param labels: Labels corresponding to the input data.
        :param save_to_path: Path where the encodings and labels will be saved.
        """
        # Reshape the data to 1D
        data = np.array(data, dtype=np.float32)  # Ensure data is a NumPy array
        data = data.reshape((data.shape[0], -1))  # Flatten each image into a 1D array

        encoder = Model(self.autoencoder.input, self.autoencoder.layers[len(self.neurons_per_layer) // 2].output)
        x = encoder.predict(data)
        x = x.astype(np.float32)
        np.save(save_to_path + '.npy', x)
        np.save(save_to_path + '_labels.npy', labels)