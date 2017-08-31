"""
This file is responsible for:
1.  Building the neural network model using tensorflow
2.  Training the model using features
3.  Testing the model
4.  Calculating Acuuracy and F1 score of the model.

This file needs a featuresets.npy file as input with extracted features.
Features can be extracted using create_feature_sets.py

@Author Sanjay Haresh Khatwani (sxk6714@rit.edu)
@Author Savitha Jayasankar (skj9180@rit.edu)
@Author Saurabh Parekh (sbp4709@rit.edu)
"""

import tensorflow as tf
import os
import numpy as np
import sklearn.metrics as sk

class TrainAndTest:
    def __init__(self):
        # Load the featuresets array
        featuresets = np.load('feature sets.npy')

        # Divide the feature sets into training and testing set.
        self.train_input, self.train_output, self.test_input, self.test_output = self.divideFeatureSets(featuresets)

        # Define number of nodes in each layer
        number_nodes_HL1 = 100
        number_nodes_HL2 = 100
        number_nodes_HL3 = 100

        # Define other constants
        n_classes = 2
        self.batch_size = 50
        self.number_epochs = 20

        # Tensorflow place holder for input ad output to the tensorflow graph
        self.x = tf.placeholder('float', [None, len(self.train_input[0])])
        self.y = tf.placeholder('float')

        # Define the layers using dictionaries. Weights and biases are initialized as
        #  random numbers.
        with tf.name_scope("HiddenLayer1"):
            self.hidden_1_layer = {'number_of_neurons': number_nodes_HL1,
                              'layer_weights': tf.Variable(
                                  tf.random_normal([len(self.train_input[0]), number_nodes_HL1])),
                              'layer_biases': tf.Variable(tf.random_normal([number_nodes_HL1]))}

        with tf.name_scope("HiddenLayer2"):
            self.hidden_2_layer = {'number_of_neurons': number_nodes_HL2,
                              'layer_weights': tf.Variable(
                                  tf.random_normal([number_nodes_HL1, number_nodes_HL2])),
                              'layer_biases': tf.Variable(tf.random_normal([number_nodes_HL2]))}

        with tf.name_scope("HiddenLayer3"):
            self.hidden_3_layer = {'number_of_neurons': number_nodes_HL3,
                              'layer_weights': tf.Variable(
                                  tf.random_normal([number_nodes_HL2, number_nodes_HL3])),
                              'layer_biases': tf.Variable(tf.random_normal([number_nodes_HL3]))}

        with tf.name_scope("OutputLayer"):
            self.output_layer = {'number_of_neurons': None,
                            'layer_weights': tf.Variable(
                                tf.random_normal([number_nodes_HL3, n_classes])),
                            'layer_biases': tf.Variable(tf.random_normal([n_classes])), }

        self.merged_summary_op = tf.summary.merge_all()
        self.logs_path = '/tmp/logs'

    def divideFeatureSets(self,features):
        """
        This method is used to divide the whole feature sets into four parts:
        1.  Training input
        2.  Training output
        3.  Testing input
        4.  Testing output

        The default split rate is 30% for testing. It can be cahnged by setting
        the value for test_size inside the method.
        :param features:
        :return: train_input, train_output, test_input, test_output.
        """
        test_size = 0.3
        testing_size = int(test_size * len(features))

        train_input = list(features[:, 0][:-testing_size])
        train_output = list(features[:, 1][:-testing_size])
        test_input = list(features[:, 0][-testing_size:])
        test_output = list(features[:, 1][-testing_size:])

        return train_input, train_output, test_input, test_output


    def neural_network_model(self,data):
        """
        This method is used to define how the data flows through the neural
        network and how inputs and outputs of different layers are calculated,
        given the feature vector.
        :param data:
        :return:
        """
        # the output of first layer is input*weights + biases
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['layer_weights']),
                    self.hidden_1_layer['layer_biases'])
        # Logit
        l1 = tf.nn.relu(l1)

        # the ouput of second layer is output of first layer *  weights + biases
        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['layer_weights']), self.hidden_2_layer['layer_biases'])
        # Logit
        l2 = tf.nn.relu(l2)

        # Similar as previous
        l3 = tf.add(tf.matmul(l2, self.hidden_3_layer['layer_weights']), self.hidden_3_layer['layer_biases'])
        l3 = tf.nn.relu(l3)

        # Finally the output is output of last layer * weights + biases. No logit
        #  for last layer as this layer's output is not fed to any other layer.
        output = tf.matmul(l3, self.output_layer['layer_weights']) + self.output_layer['layer_biases']

        return output


    def train_neural_network(self):
        """
        This method is responsible for training the NN model using
        backpropagation and tensorflow
        """

        # Get the prediction.
        with tf.name_scope("model"):
            prediction = self.neural_network_model(self.x)

        # Get the cost of this prediction, which is passed through softmax and
        # then reduced mean is computed to give final cost.
        with tf.name_scope("Cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                  (logits=prediction, labels=self.y))

        # The goal of back-propagation is to minimize the cost. Use AdamOptimizer
        #  for that.
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        tf.summary.scalar("COST", cost)

        merged_summary_op = tf.summary.merge_all()

        # Time to trigger Tensor flow
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(self.logs_path,
                                                   graph=tf.get_default_graph())

            # Train in epochs
            for epoch in range(self.number_epochs):
                epoch_loss = 0
                i = 0
                while i < len(self.train_input):
                    start = i
                    end = i + self.batch_size

                    # Divide into batches
                    batch_x = np.array(self.train_input[start:end])
                    batch_y = np.array(self.train_output[start:end])

                    # Run tensorflow to train using this batch
                    _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                             feed_dict={self.x: batch_x, self.y: batch_y})
                    summary_writer.add_summary(summary, epoch * self.batch_size + i)
                    # Keep aggregating the cost/loss for calculating the loss of
                    # the whole epoch
                    epoch_loss += c

                    # Increment the batch-start index
                    i += self.batch_size

                print('Epoch', epoch + 1, 'completed out of', self.number_epochs, 'loss:',
                      epoch_loss)

            # Specify the correctness criteria. Prediction = actual
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

            # Calculate accuracies correctness criteria
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # Specify the criteria to get predicted classes, with a default of 1.
            y_p = tf.argmax(prediction, 1)

            # Run the model on test data to obtain predictions
            val_accuracy, y_pred = sess.run([accuracy, y_p],
                                            feed_dict={self.x: self.test_input,
                                                       self.y: self.test_output})
            # Get actual classes
            y_true = np.argmax(self.test_output, 1)

            #Calculate f1 score using scikit-learn.
            print('F1 Score:', sk.f1_score(y_true, y_pred))

            #Print out the confusion matrix
            print(sk.confusion_matrix(y_true, y_pred))

            print('Accuracy:', accuracy.eval({self.x: self.test_input, self.y: self.test_output}))

            # Save the model for using in future.
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(os.getcwd(), 'model/sarcasm_model'))


if __name__ == '__main__':

    TrainAndTest().train_neural_network()
    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
