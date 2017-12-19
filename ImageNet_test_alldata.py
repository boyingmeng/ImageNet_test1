import numpy as np
# import matplotlib as mp
# matplotlib inline
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.examples.tutorials.mnist import input_data
import math
import time
import pickle
import os
#import dataset

# import lasagne

start_time = time.time()

tf.reset_default_graph()

img_size_0 = 16
n_class = 1000

# using downsized imagenet from
# https://patrykchrabaszcz.github.io/Imagenet32/

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

# from mnist
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def load_databatch(data_folder, img_size):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    idx = 1
    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    #x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    for idx in range(9):
        d = unpickle(data_file + str(idx+2))
        x1 = d['data']
        y1 = d['labels']
        mean_image1 = d['mean']

        x1 = x1 / np.float32(255)
        mean_image1 = mean_image1 / np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y1 = [i - 1 for i in y1]
        data_size += x1.shape[0]

        x1 -= mean_image1

        img_size2 = img_size * img_size

        x1 = np.dstack((x1[:, :img_size2], x1[:, img_size2:2 * img_size2], x1[:, 2 * img_size2:]))
        # x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        x1 = x1.reshape(x1.shape[0], img_size, img_size, 3)

        x = np.concatenate((x, x1), axis=0)
        y = np.concatenate((y, y1), axis=0)



    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    Y_train = dense_to_one_hot(Y_train,n_class)

    # validation data
    data_file = os.path.join(data_folder, 'val_data')

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    # mean_image = d['mean']

    x = x / np.float32(255)
    # mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]

    # x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    # x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    X_val = x[0:data_size, :, :, :]
    Y_val = np.array(y[0:data_size])
    Y_val = dense_to_one_hot(Y_val,n_class)


    return dict(
        # X_train=lasagne.utils.floatX(X_train),
        X_train=X_train,
        #Y_train=Y_train.astype('int32'),
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        mean=mean_image)


imagenet = load_databatch('../Imagenet' + str(img_size_0) + '_train/', img_size_0)

#for i in range(9):
#    imagenet1 = load_databatch('Imagenet' + str(img_size_0) + '_train/', i+2, img_size_0)
#    imagenet['X_train'] = np.concatenate((imagenet['X_train'],imagenet1['X_train']), axis=0)
#    imagenet['Y_train'] = np.concatenate((imagenet['Y_train'],imagenet1['Y_train']), axis=0)


def next_batch(num, data, labels):

    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    #labels_shuffle = np.asarray(labels_shuffle.values.reshape(len(labels_shuffle), 1))

    return data_shuffle, labels_shuffle


# define the nn
x = tf.placeholder(tf.float32, [None, img_size_0, img_size_0, 3], name="x-in")
true_y = tf.placeholder(tf.float32, [None, n_class], name="y-in")
keep_prob = tf.placeholder("float")

#x_image = tf.reshape(x, [-1, 28, 28, 1])  # reshape
net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
net = slim.conv2d(net, 192, [5, 5], scope='conv2')
#net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
net = slim.conv2d(net, 384, [3, 3], scope='conv3')
net = slim.conv2d(net, 384, [3, 3], scope='conv4')
net = slim.conv2d(net, 256, [3, 3], scope='conv5')
#net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

# hidden_3 = slim.dropout(hidden_3,keep_prob)    # dropout
# fc 1x
net = slim.fully_connected(slim.flatten(net), 2048, scope='fc1')
net = slim.fully_connected(slim.flatten(net), 2048, scope='fc2')
net = slim.dropout(net, keep_prob, scope='dropout')  # dropout
# out_y = slim.fully_connected(slim.flatten(h_fc1), 10, activation_fn=tf.nn.softmax,scope='fc2')
out_y = slim.fully_connected(slim.flatten(net), n_class, scope='fc3')

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=true_y,
                                                            logits=out_y)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(true_y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

batchSize = 2000
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(200001):
    batch = next_batch(batchSize, imagenet['X_train'], imagenet['Y_train'])
    sess.run(train_step, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 0.5})
    # if i % 100 == 0 and i != 0:
    if i % 10 == 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
        #print('%d' % (i))
        #print('%g' % (trainAccuracy))
        if i % 10 == 0:
            testAccuracy = sess.run(accuracy, feed_dict={x: imagenet['X_val'], true_y: imagenet['Y_val'], keep_prob: 1.0})
            print("test accuracy %g" % (testAccuracy))


#testAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, true_y: mnist.test.labels, keep_prob: 1.0})
#print("test accuracy %g" % (testAccuracy))

print("time elapsed: {:.2f}s".format(time.time() - start_time))
