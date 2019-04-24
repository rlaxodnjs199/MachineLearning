from pandas.io.parsers import read_csv
import numpy as np
import tensorflow as tf

data = read_csv('data.csv', sep=',')
tf.global_variables_initializer()
arr = np.array(data, dtype=np.float32)
x_data = arr[:, 1:-1]
y_data = arr[:, [-1]]
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000010)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100001):
    cost, hypo, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print("#", step, " Cost: ", cost)
        print("Price: ", hypo[0])
saver = tf.train.Saver()
save_path = saver.save(sess, "./saved.cpkt")