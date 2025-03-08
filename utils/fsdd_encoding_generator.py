from free_spoken_digit_dataset.utils.fsdd import FSDD
from cdzproject.modules.autoencoder.autoencoder import Autoencoder


def generate_encodings():
    """
    Generates encodings for the Free Spoken Digit Dataset (FSDD).
    """
    # Load spectrograms and labels from the FSDD dataset
    images, labels = FSDD.get_spectrograms()
    labels = [int(label) for label in labels]  # Ensure labels are integers

    # Split into train and test sets
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for idx in range(10):
        # Extract training dataset
        start = idx * 50 + 5
        end = start + 45
        train_images.extend(images[start:end])
        train_labels.extend(labels[start:end])

        # Extract testing dataset
        start = idx * 50
        end = start + 5
        test_images.extend(images[start:end])
        test_labels.extend(labels[start:end])

    # Initialize the autoencoder
    # Many epochs are used because the training set is small and the learning rate is low.
    autoencoder = Autoencoder(
        neurons_per_layer=[4096, 256, 64],
        pretrain=True,
        pretrain_epochs=20,
        finetune_epochs=500,
        finetune_batch_size=16
    )
    autoencoder.train(train_images)

    # Generate and save encodings for the train and test datasets
    autoencoder.generate_encodings(
        train_images, train_labels, save_to_path='../data/fsdd_train_encodings_2'
    )
    autoencoder.generate_encodings(
        test_images, test_labels, save_to_path='../data/fsdd_test_encodings_2'
    )


if __name__ == '__main__':
    generate_encodings()