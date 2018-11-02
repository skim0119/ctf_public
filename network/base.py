class bass():
    def __init__(self):
        pass

    def build_train(self):
        pass

    def build_network(self):
        pass

    def build_summarizer(self):
        pass

    #Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(std=1.0):
            def _initializer(shape, dtype=None, partition_info=None):
                        out = np.random.randn(*shape).astype(np.float32)
                                out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                                        return tf.constant(out)
                                        return _initializer
