# Our Custom L1 Distance Layer Module 


import tensorflow as tf
from tensorflow.keras.layers import Layer

#From Jupyter
#Siamese Distance Class
class L1Dist(Layer):

    #Init Inheritience Method
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def call(self,inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)