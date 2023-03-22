
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, voc_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(voc_size, embedding_dim)
        self.batch_size = batch_size
        self.encoder_units = encoder_units       
        self.gru = tf.keras.layers.GRU(self.encoder_units,                                   
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x,hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))