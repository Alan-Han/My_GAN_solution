#!/usr/bin/python3
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from My_GAN_solution import input_dataset
from My_GAN_solution import gan


# There are two ways of solving this problem.
# One is to have the matmul at the last layer output all 11 classes.
# The other is to output just 10 classes, and use a constant value of 0 for
# the logit for the last class. This still works because the softmax only needs
# n independent logits to specify a probability distribution over n + 1 categories.
# We implemented both solutions here.
extra_class = 0



def train(net, dataset, epochs, batch_size, z_size=100, figsize=(5, 5)):
    saver = tf.train.Saver()
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    samples, train_accuracies, test_accuracies = [], [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(1, epochs):
            print("Epoch", e)

            t1e = time.time()
            num_examples = 0
            num_correct = 0
            for x, y, label_mask in dataset.batches(batch_size):
                assert 'int' in str(y.dtype)
                steps += 1
                num_examples += label_mask.sum()

                # Sample random noise for G
                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                # Run optimizers
                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y: y, net.label_mask: label_mask})
                t2 = time.time()
                num_correct += correct

            sess.run([net.shrink_lr])

            train_accuracy = num_correct / float(num_examples)

            print("\t\tClassifier train accuracy: ", train_accuracy)

            num_examples = 0
            num_correct = 0
            for x, y in dataset.batches(batch_size, which_set="test"):
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]

                correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                              net.y: y,
                                                              net.drop_rate: 0.})
                num_correct += correct

            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)

            gen_samples = sess.run(
                net.samples,
                feed_dict={net.input_z: sample_z})
            samples.append(gen_samples)
            _ = input_dataset.view_samples(-1, samples, 5, 10, figsize=figsize)
            plt.show()

            # Save history of accuracies to view after training
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        saver.save(sess, './checkpoints/generator.ckpt')

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return train_accuracies, test_accuracies, samples




def main():
    class Dataset:
        def __init__(self, train, test, val_frac=0.5, shuffle=True, scale_func=None):
            split_idx = int(len(test['y']) * (1 - val_frac))
            self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
            self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
            self.train_x, self.train_y = train['X'], train['y']
            # The SVHN dataset comes with lots of labels, but for the purpose of this exercise,
            # we will pretend that there are only 1000.
            # We use this mask to say which labels we will allow ourselves to use.
            self.label_mask = np.zeros_like(self.train_y)
            self.label_mask[0:1000] = 1

            self.train_x = np.rollaxis(self.train_x, 3)
            self.valid_x = np.rollaxis(self.valid_x, 3)
            self.test_x = np.rollaxis(self.test_x, 3)

            if scale_func is None:
                self.scaler = input_dataset.scale
            else:
                self.scaler = scale_func
            self.train_x = self.scaler(self.train_x)
            self.valid_x = self.scaler(self.valid_x)
            self.test_x = self.scaler(self.test_x)
            self.shuffle = shuffle

        def batches(self, batch_size, which_set="train"):
            x_name = which_set + "_x"
            y_name = which_set + "_y"

            num_examples = len(getattr(dataset, y_name))
            if self.shuffle:
                idx = np.arange(num_examples)
                np.random.shuffle(idx)
                setattr(dataset, x_name, getattr(dataset, x_name)[idx])
                setattr(dataset, y_name, getattr(dataset, y_name)[idx])
                if which_set == "train":
                    dataset.label_mask = dataset.label_mask[idx]

            dataset_x = getattr(dataset, x_name)
            dataset_y = getattr(dataset, y_name)
            for ii in range(0, num_examples, batch_size):
                x = dataset_x[ii:ii + batch_size]
                y = dataset_y[ii:ii + batch_size]

                if which_set == "train":
                    # When training, we need to include label mask
                    yield x, y, self.label_mask[ii:ii + batch_size]
                else:
                    yield x, y

    trainset, testset = input_dataset.download_data()

    real_size = (32, 32, 3)
    z_size = 100
    learning_rate = 0.01  #0.0003

    net = gan.GAN(real_size, z_size, learning_rate)

    dataset = Dataset(trainset, testset)

    batch_size = 64  # 128
    epochs = 25
    train_accuracies, test_accuracies, samples = train(net, dataset, epochs, batch_size, figsize=(10, 5))

    fig, ax = plt.subplots()
    plt.plot(train_accuracies, label='Train', alpha=0.5)
    plt.plot(test_accuracies, label='Test', alpha=0.5)
    plt.title("Accuracy")
    plt.legend()


if __name__ == '__main__':
    main()