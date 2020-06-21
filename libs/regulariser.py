import tensorflow as tf


class ShiftedL1L2(tf.keras.regularizers.L1L2):
    """L1L2 Regularizer, but the input is shifted, i.e. zero loss occurs at the given point"""
    def __init__(self, new_zero: float, l1=0., l2=0.):
        super(ShiftedL1L2, self).__init__(l1=l1, l2=l2)

        self.new_zero = new_zero
        self.__name__ = "ShiftedL1L2"

    def __call__(self, x):
        return super(ShiftedL1L2, self).__call__(x=x-self.new_zero)

    def get_config(self):
        config = {
            'new_zero': self.new_zero,
        }
        base_config = super(ShiftedL1L2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
