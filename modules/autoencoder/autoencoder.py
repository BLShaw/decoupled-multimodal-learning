import numpy as np
import tensorflow as tf
from yadlt.models.autoencoder_models.deep_autoencoder import DeepAutoencoder


class Autoencoder:

    def __init__(self, neurons_per_layer, pretrain, pretrain_epochs, finetune_epochs, finetune_batch_size):
        """
        A wrapper for a yadlt deep autoencoder.

        :param neurons_per_layer: List of integers specifying the number of neurons in each layer.
        :param pretrain: Boolean indicating whether to perform pretraining.
        :param pretrain_epochs: Number of epochs for pretraining.
        :param finetune_epochs: Number of epochs for fine-tuning.
        :param finetune_batch_size: Batch size for fine-tuning.
        """
        self.autoencoder_config = {
            'layers': neurons_per_layer,
            'enc_act_func': [tf.nn.sigmoid],
            'dec_act_func': [tf.nn.sigmoid],
            'finetune_num_epochs': finetune_epochs,
            'finetune_loss_func': 'mean_squared',
            'finetune_dec_act_func': [tf.nn.sigmoid],
            'finetune_enc_act_func': [tf.nn.sigmoid],
            'finetune_opt': 'adam',
            'finetune_learning_rate': 1e-4,
            'finetune_batch_size': finetune_batch_size,
            'do_pretrain': pretrain,
            'num_epochs': [pretrain_epochs],
            'verbose': 1,
            'corr_frac': [.5],
            'corr_type': ["masking"]
        }

        self.encoder = DeepAutoencoder(**self.autoencoder_config)

    def train(self, data):
        """
        Trains the autoencoder on the given data.

        :param data: Input data to train the autoencoder.
        """
        # Normalize the data
        # We need to do this because we are using sigmoid that has a range of [0, 1]
        data = np.array([blurb / max(abs(blurb)) if max(abs(blurb)) != 0 else blurb for blurb in data])
        np.random.shuffle(data)

        if self.autoencoder_config['do_pretrain']:
            self.encoder.pretrain(data, validation_set=data)

        self.encoder.fit(data, data)

    def generate_encodings(self, data, labels, save_to_path):
        """
        Generates encodings for the input data and saves them to the specified path.

        :param data: Input data to encode.
        :param labels: Labels corresponding to the input data.
        :param save_to_path: Path where the encodings and labels will be saved.
        """
        x = self.encoder.transform(data)
        x = x.astype(np.float32)
        np.save(save_to_path + '.npy', x)
        np.save(save_to_path + '_labels.npy', labels)