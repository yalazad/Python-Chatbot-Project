import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, voc_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(voc_size, embedding_dim)
        self.encoder_units = encoder_units
        self.batch_size = batch_size
              
        self.lstm = tf.keras.layers.LSTM(
                                            self.encoder_units,
                                            dropout = 0.22,                                    
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform'
                                        )

    def call(self, x, hidden):
        x = self.embedding(x)
        dim = tf.zeros([self.batch_size, self.encoder_units])     
        output, hidden_state, cell_state = self.lstm(x, initial_state = [dim, dim])

        return output, hidden_state, cell_state
    
    def initialize_hidden_state(self):
        return tf.zeros([self.batch_size, self.encoder_units])