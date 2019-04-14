""" NN Model Wrapper """

import numpy as np

import tensorflow as tf

from utility.utils import store_args


class TrainedNetwork:
    @store_args
    def __init__(
        self,
        model_name,
        input_tensor='global/state:0',
        output_tensor='global/actor/fully_connected_1/Softmax:0',
        action_space=5,
        *args,
        **kwargs
    ):
        self.model_path = 'model/' + model_name

        # Initialize Session and TF graph
        self.initialize_network()

    def get_action(self, input_tensor):
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state: input_tensor}
            action_prob = self.sess.run(self.action, feed_dict)

        action_out = [np.random.choice(self.action_space, p=prob / sum(prob)) for prob in action_prob]

        return action_out

    def reset_network_weight(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise AssertionError

    def initialize_network(self, verbose=False):
        """reset_network
        Initialize network and TF graph
        """
        def vprint(*args):
            if verbose:
                print(args)

        input_tensor = self.input_tensor
        output_tensor = self.output_tensor

        config = tf.ConfigProto(device_count={'GPU': 0})

        # Reset the weight to the newest saved weight.
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        vprint(f'path find: {ckpt.model_checkpoint_path}')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            vprint(f'path exist : {ckpt.model_checkpoint_path}')
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf.Session(config=config)
                self.saver = tf.train.import_meta_graph(
                    ckpt.model_checkpoint_path + '.meta',
                    clear_devices=True
                )
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                vprint([n.name for n in self.graph.as_graph_def().node])

                self.state = self.graph.get_tensor_by_name(input_tensor)

                try:
                    self.action = self.graph.get_operation_by_name(output_tensor)
                except ValueError:
                    self.action = self.graph.get_tensor_by_name(output_tensor)
                    vprint([n.name for n in self.graph.as_graph_def().node])

            vprint('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            vprint('Error : Graph is not loaded')
            raise NameError
