import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import text_preprocessor as pre
import lstm_encoder as lstm_encoder
import lstm_decoder as lstm_decoder
import bahdanau_attention as bahdanau
import datetime
import os
          
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(input, resp, encoder_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        encoder_output, encoder_hidden, encoder_cell = encoder(input, encoder_hidden)

        decoder_hidden = encoder_hidden

        decoder_input = tf.expand_dims([responses.word_index['<sos>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, resp.shape[1]):
            # passing encoder output to the decoder
            predictions, decoder_hidden, _, _ = decoder(decoder_input, decoder_hidden, encoder_cell, encoder_output)

            loss += loss_function(resp[:, t], predictions)

            # using teacher forcing
            decoder_input = tf.expand_dims(resp[:, t], 1)
        
            train_loss(loss)
            train_accuracy(resp[:, t], predictions)

    batch_loss = (loss / int(resp.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

@tf.function
def test_step(val_inp, val_targ, encoder_hidden):
    loss = 0
    encoder_output, encoder_hidden, encoder_cell = encoder(val_inp, encoder_hidden)

    decoder_hidden = encoder_hidden
    
    decoder_input = tf.expand_dims([responses.word_index['<sos>']] * BATCH_SIZE, 1)

    # Teacher forcing - feed the target as the next input
    for t in range(1, val_targ.shape[1]):
        # passing the encoder output to the decoder
        predictions, decoder_hidden, _, _ = decoder(decoder_input, decoder_hidden, encoder_cell, encoder_output)

        loss += loss_function(val_targ[:, t], predictions)

        # using Teacher forcing
        decoder_input = tf.expand_dims(val_targ[:, t], 1)
    
        val_loss(loss)
        val_accuracy(val_targ[:, t], predictions)

    batch_loss = (loss / int(val_targ.shape[1]))

    return batch_loss

def evaluate(sentence):
    print("IN EVALUATE!")     

    sentence = pre.clean_text(sentence)

    enc_inputs = [inputs.word_index[i] for i in sentence.split(' ')]
    enc_inputs = tf.keras.preprocessing.sequence.pad_sequences([enc_inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    enc_inputs = tf.convert_to_tensor(enc_inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    encoder_out, encoder_hidden, encoder_cell = encoder(enc_inputs, hidden)

    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([responses.word_index['<sos>']], 0)

    for t in range(max_length_resp):
        predictions, decoder_hidden, encoder_cell, attention_weights = decoder(decoder_input,
                                                             decoder_hidden,
                                                             encoder_cell,
                                                             encoder_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += responses.index_word[predicted_id] + ' '

        if responses.index_word[predicted_id] == '<eos>':
            return pre.remove_tags(result), pre.remove_tags(sentence)

        # feed the predicted ID back into the model
        decoder_input = tf.expand_dims([predicted_id], 0)

    return pre.remove_tags(result), pre.remove_tags(sentence)

questions = []
answers = []
with open("./data/q_and_a_london.txt",'r') as f :
    for line in f :
        line  =  line.split('\t')
        questions.append(line[0])
        answers.append(line[1])

answers = [ i.replace("\n","") for i in answers]

questions = [pre.clean_text(i) for i in questions]
answers = [pre.clean_text(i) for i in answers]

input_tensor, inputs  =  pre.tokenize(questions)
response_tensor, responses  =  pre.tokenize(answers)

max_length_resp, max_length_inp = response_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, response_tensor_train, response_tensor_val = train_test_split(input_tensor, response_tensor, test_size=0.3)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 16

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
val_steps_per_epoch = len(input_tensor_val)//BATCH_SIZE
embedding_dim = 256
units = 256
vocab_inp_size = len(inputs.word_index)+1
vocab_resp_size = len(responses.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, response_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, response_tensor_val)).shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

# Create encoder
encoder = lstm_encoder.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# Create attention layer
attention_layer = bahdanau.BahdanauAttention(10)

# Create decoder
decoder = lstm_decoder.Decoder(vocab_resp_size, embedding_dim, units, BATCH_SIZE)

# Create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Checkpoints
checkpoint_dir = './checkpoints_lstm'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  
  
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

EPOCHS = 100

# For early stopping
patience = 5
wait = 0
best = float('inf')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = current_time + '/lstm/train'
val_log_dir = current_time + '/lstm/val'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

if manager.latest_checkpoint:
    print("CHECKPOINT FOUND!")
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(manager.latest_checkpoint)
else:    
    for epoch in range(1, EPOCHS + 1):
        encoder_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        val_total_loss = 0

        for (batch, (inp, resp)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, resp, encoder_hidden)
            total_loss += batch_loss

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
    
        for (val_batch, (val_inp, val_resp)) in enumerate(val_dataset.take(val_steps_per_epoch)):
            val_batch_loss = test_step(val_inp, val_resp, encoder_hidden)
            val_total_loss += val_batch_loss

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        # The early stopping strategy: stop the (training) if `val_loss` does not
        # decrease over a certain number of epochs.       
        """
        wait += 1
        if val_loss.result() < best:
            best = val_loss.result()
            wait = 0
        if wait >= patience:
            break  
        """          
    
        if(epoch % 4 == 0):
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('Epoch:{:3d} Loss:{:.4f}'.format(epoch,
                                            total_loss / steps_per_epoch))
        
        # Reset metrics every epoch
        train_loss.reset_states()
        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()            