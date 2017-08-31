"""
This file is used to use the predict the trained neural network and get
predictions for new input. The input an be given in the last line while
calling use_neural_network() method.

@Author Sanjay Haresh Khatwani (sxk6714@rit.edu)
@Author Savitha Jayasankar (skj9180@rit.edu)
@Author Saurabh Parekh (sbp4709@rit.edu)
"""

from createFeatureSets import CreateFeatureSet
import tensorflow as tf
import os


class sarcasm_test:
    # Build the structure of the neural network exactly same as the
    # trainAndTest.py, so that the input features can be run through the neural
    #  network.
    def __init__(self):
        number_nodes_HL1 = 100
        number_nodes_HL2 = 100
        number_nodes_HL3 = 100

        self.x = tf.placeholder('float', [None, 23])
        self.y = tf.placeholder('float')

        with tf.name_scope("HiddenLayer1"):
            self.hidden_1_layer = {'number_of_neurons': number_nodes_HL1,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal([23, number_nodes_HL1])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL1]))}

        with tf.name_scope("HiddenLayer2"):
            self.hidden_2_layer = {'number_of_neurons': number_nodes_HL2,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal(
                                           [number_nodes_HL1, number_nodes_HL2])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL2]))}

        with tf.name_scope("HiddenLayer3"):
            self.hidden_3_layer = {'number_of_neurons': number_nodes_HL3,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal(
                                           [number_nodes_HL2, number_nodes_HL3])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL3]))}

        with tf.name_scope("OutputLayer"):
            self.output_layer = {'number_of_neurons': None,
                                 'layer_weights': tf.Variable(
                                     tf.random_normal([number_nodes_HL3, 2])),
                                 'layer_biases': tf.Variable(tf.random_normal([2])), }
        self.saver = tf.train.Saver()

    # Nothing changes in this method as well.
    def neural_network_model(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['layer_weights']),
                    self.hidden_1_layer['layer_biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['layer_weights']),
                    self.hidden_2_layer['layer_biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, self.hidden_3_layer['layer_weights']),
                    self.hidden_3_layer['layer_biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, self.output_layer['layer_weights']) + self.output_layer[
            'layer_biases']

        return output

    def use_neural_network(self, input_data):
        """
        In this method we restore the model created previously and obtain a
        prediction for an input sentence.
        :param input_data:
        :return:
        """
        prediction = self.neural_network_model(self.x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, os.path.join(os.getcwd(),
                                                  'model/sarcasm_model'))
            features = CreateFeatureSet().extract_feature_of_sentence(input_data)
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={self.x: [
                features]}), 1)))
            if result[0] == 0:
                print('Sarcastic:', input_data)
            elif result[0] == 1:
                print('Regular:', input_data)


# Supply the sentence to be tested below as a parameter in the method call.
if __name__ == '__main__':
    sarcasm_test().use_neural_network("Going to the gym surely makes you fit")
