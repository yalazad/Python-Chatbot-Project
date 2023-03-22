import tensorflow as tf
import bahdanau_attention as ba

class Decoder(tf.keras.Model):
    def __init__(self, voc_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(voc_size, embedding_dim)
        self.decoder_units = decoder_units
        self.batch_size = batch_size
        
        self.lstm = tf.keras.layers.LSTM(
                                            self.decoder_units,
                                            dropout = 0.22,                                            
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform'
                                        )        
        self.fc = tf.keras.layers.Dense(voc_size)

        # used for attention
        self.attention = ba.BahdanauAttention(self.decoder_units)

    def call(self, x, hidden, cell_state, encoder_output):
        # shape of encoder_output == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, encoder_output)

        # shape of x after passing through the embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # shape of x after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # the concatenated vector is passed to the LSTM
        output, hidden_state, cell_state = self.lstm(x)

        # the output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, hidden_state, cell_state, attention_weights