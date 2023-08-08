import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from matplotlib import pyplot as plt

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

figure = plt.figure(figsize=(10, 5))
j=0

for example in example_dataset.take(20):
  plt.subplot(5, 5, j+1)
  plt.imshow(example['pixels'].numpy(), cmap='gray', aspect='equal')
  plt.axis('on')
  j += 1
