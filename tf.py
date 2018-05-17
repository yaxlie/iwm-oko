import tensorflow as tf
import glob
import random
import skimage.io
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
 
 
NUM_ITERS=500
DISPLAY_STEP=10
BATCH=10
tf.set_random_seed(0)
 
# Download images and labels 
#mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)
# mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels<br># mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels
X = tf.placeholder(tf.float32, [None, 64, 64, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 1])
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)


images=[]
clasess=[]
for img in glob.glob("fragments64/0true/*.jpg"):
	images.append(skimage.io.imread(img).astype(np.float32).tolist())
	clasess.append([1.0])
for img in glob.glob("fragments64/0false/*.jpg"):
	images.append(skimage.io.imread(img).astype(np.float32).tolist())
	clasess.append([0.0])
X=images
Y_=np.asarray(clasess).astype(np.float32).tolist()
#Y_=tf.reshape(Y_, [-1, 1])

im=[]
cl=[]
for img in glob.glob("fragments64(2)/0true/*.jpg"):
	im.append(skimage.io.imread(img).astype(np.float32).tolist())
	cl.append(1.0)
for img in glob.glob("fragments64(2)/0false/*.jpg"):
	im.append(skimage.io.imread(img).astype(np.float32).tolist())
	cl.append(0.0)
testIm=im
testLb=cl
testLb=tf.reshape(testLb, [-1, 1])

# layers sizes
C1 = 3  # first convolutional layer output depth
C2 = 8  # second convolutional layer output depth
C3 = 16 # third convolutional layer output depth
 
FC4 = 256  # fully connected layer
 
 
# weights - initialized with random values from normal distribution mean=0, stddev=0.1
 
# 5x5 conv. window, 1 input channel (gray images), C1 - outputs
W1 = tf.Variable(tf.truncated_normal([5, 5, 3, C1], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([C1], stddev=0.1))
# 3x3 conv. window, C1 input channels(output from previous conv. layer ), C2 - outputs
W2 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([C2], stddev=0.1))
# 3x3 conv. window, C2 input channels(output from previous conv. layer ), C3 - outputs
W3 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([C3], stddev=0.1))
# fully connected layer, we have to reshpe previous output to one dim, 
# we have two max pool operation in our network design, so our initial size 28x28 will be reduced 2*2=4
# each max poll will reduce size by factor of 2
W4 = tf.Variable(tf.truncated_normal([8*8*C3, FC4], stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.1))
 
# output softmax layer (10 digits)
W5 = tf.Variable(tf.truncated_normal([FC4, 1], stddev=0.1))
b5 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
 
#XX = tf.reshape(X, [-1, 12288])
 
# Define the model
 
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)
 
k = 4 # max pool filter size and stride, will reduce input by factor of 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

k = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)
Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
 
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 8 * 8 * C3])
 
Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
#Y4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
 
                                                          
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# training, 
learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# matplotlib visualization
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)

# Initializing the variables
init = tf.global_variables_initializer()
 
train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()
 
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
 
 
    for i in range(NUM_ITERS+1):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = tf.train.batch([X, Y_], batch_size = BATCH)
        
        if i%DISPLAY_STEP ==0:
            # compute training values for visualisation
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
            
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: testIm, Y_: testLb, pkeep: 1.0})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))
 		
            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_trn)
            test_acc.append(acc_tst)
 
        # the backpropagationn training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})
