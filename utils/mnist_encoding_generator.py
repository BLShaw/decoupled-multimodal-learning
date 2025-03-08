import numpy as np
from yadlt.utils import datasets
from cdzproject.modules.autoencoder.autoencoder import Autoencoder


def generate_encodings():
    """
    Generates encodings for the MNIST dataset.
    """
    # Load the MNIST dataset in supervised mode
    (
        train_images, train_labels,
        validation_images, validation_labels,
        test_images, test_labels
    ) = datasets.load_mnist_dataset(mode='supervised')

    # Convert one-hot encoded labels to integers
    train_labels = [np.argmax(label) for label in train_labels]
    test_labels = [np.argmax(label) for label in test_labels]

    # Initialize the autoencoder
    autoencoder = Autoencoder(
        neurons_per_layer=[2048, 1024, 256, 128],
        pretrain=False,
        pretrain_epochs=0,
        finetune_epochs=120,
        finetune_batch_size=64
    )
    autoencoder.train(train_images)

    # Generate and save encodings for the train and test datasets
    autoencoder.generate_encodings(
        train_images, train_labels, save_to_path='../data/mnist_train_encodings_6'
    )
    autoencoder.generate_encodings(
        test_images, test_labels, save_to_path='../data/mnist_test_encodings_6'
    )


if __name__ == '__main__':
    generate_encodings()