"""
This is the dataset used in the paper.
"""

import random
from collections import defaultdict
import numpy as np

# Load visual data
v_train_data = np.load('../data/mnist_train_encodings.npy')
v_train_labels = np.load('../data/mnist_train_encodings_labels.npy')

v_test_data = np.load('../data/mnist_test_encodings.npy')
v_test_labels = np.load('../data/mnist_test_encodings_labels.npy')

# Load audio data
a_train_data = np.load('../data/fsdd_train_encodings.npy')
a_train_labels = np.load('../data/fsdd_train_encodings_labels.npy')

a_test_data = np.load('../data/fsdd_test_encodings.npy')  # Fixed typo: 'fsdd_test_encodingsnpy' → 'fsdd_test_encodings.npy'
a_test_labels = np.load('../data/fsdd_test_encodings_labels.npy')

# Create a dictionary to map labels to audio encodings
audio_dict = defaultdict(list)
for idx, label in enumerate(a_train_labels):
    audio_dict[str(label)].append(a_train_data[idx])


def get_random_train_data():
    """
    Retrieves a random training example consisting of a visual encoding, audio encoding, and label.

    :return: A tuple containing the visual encoding, audio encoding, and label.
    """
    rand_idx = random.randint(0, len(v_train_data) - 1)
    visual_encoding = v_train_data[rand_idx]
    label = v_train_labels[rand_idx]

    # Get a random audio example of the same label
    rand_idx_audio = random.randint(0, len(audio_dict[str(label)]) - 1)
    audio_encoding = audio_dict[str(label)][rand_idx_audio]

    return visual_encoding, audio_encoding, np.float32(label)