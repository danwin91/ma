import numpy as np 
import tensorflow as tf

shape = (20000,2000)
BIG_MATRIX = np.random.random(size=shape)
print(BIG_MATRIX.shape)


import time 
start = time.time()
print("starting np")
np.linalg.svd(BIG_MATRIX)
end = time.time()
print("end np, time elapsed {}".format(end - start))
#
m = tf.placeholder(tf.float32, shape=shape)
tf_svd = tf.svd(m, compute_uv = True)
print("starting tf")
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    svd, u, v = sess.run(tf_svd, feed_dict={m:BIG_MATRIX})
end = time.time()
print("end tf, time elapsed {}".format(end - start))


