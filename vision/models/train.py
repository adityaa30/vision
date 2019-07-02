import time
import numpy as np
import tensorflow as tf

from vision.models.models import *
from vision.coco import COCODataset

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(image_tensor, target, encoder, decoder, dataset, optimizer):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([dataset.tokenizer.word_index['<start>']] * dataset.batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(image_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

def train():
    dataset = COCODataset()

    embedding_dim = 256
    units = 512
    vocab_size = len(dataset.tokenizer.word_index) + 1
    num_steps = len(dataset.train_captions) // dataset.batch_size
    # Shape of the vector extracted from MobileNet is (49, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 49

    encoder = CNNEncoder(embedding_dim)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    # Checkpoints
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    EPOCHS = 20

    print('Starting training..')
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset.train_dataset):
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, dataset, optimizer)
            total_loss += t_loss

            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 2 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    np.save('loss_plot.npy', np.array(loss_plot))
