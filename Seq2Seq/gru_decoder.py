
import tensorflow as tf
import bahdanau_attention as ba

class Decoder(tf.keras.Model):
    def __init__(self, voc_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(voc_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.decoder_units,                                                                  
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(voc_size)

        # used for attention
        self.attention = ba.BahdanauAttention(self.decoder_units)

    def call(self, x, hidden, encoder_output):
        # shape of enc_output == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, encoder_output)

        # shape of x after passing through the embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # shape of x after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing of the concatenated vector to the GRU
        output, state = self.gru(x)

        # the output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # the output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights