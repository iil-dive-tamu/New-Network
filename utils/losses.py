from tensorflow.keras import backend as K
from tensorflow.keras.backend import categorical_crossentropy
import tensorflow as tf



def binary_crossentropy(y_true, y_pred):
    """
    Simple binary_crossentropy application
    """
    return  K.mean(K.binary_crossentropy(y_true, y_pred))


