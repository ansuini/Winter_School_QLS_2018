# Third-party libraries
import tensorflow as tf

  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

filter_shape=(1,10,5,5)
pool_size=(2,2)
strides=2
num_epochs=5

input_layer= tf.reshape(train_data["x"], [-1, 28, 28, 1])
conv = tf.layers.conv2d(
          inputs=input_layer,
          filters=filter_shape[0],
          kernel_size=[filter_shape[2], filter_shape[3]],
          padding="same",
          activation=tf.nn.relu)
pool = tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size, strides=strides) 
pool_flat = tf.reshape(pool, [-1, 14 * 14 * 10])
logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

    
      loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        
        
        
                for epoch in range(num_epochs):
            
            _,epoch_cost=sess.run([optimizer, loss], feed_dict={X: x_train, Y: y_train})
            
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        params = sess.run(params)
        print("Parameters have been trained!")
        y_predicted=sess.run([Y_predicted], feed_dict={X: x_train, Y: y_train})
        y_predicted_test=sess.run([Y_predicted], feed_dict={X: x_test, Y: y_test})
