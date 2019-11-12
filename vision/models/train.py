import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf

from vision.flickr import FlickrDataset, load_image
from vision.models.models import (
    MobileNetV2, BahdanauAttention, CNNEncoder, RNNDecoder)

# Shape of the vector extracted from MobileNet is (49, 2048)
# These two variables represent that vector shape
FEATURE_SHAPE = 2048
ATTENTION_FEATURE_SHAPE = 49
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 30


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
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

    dec_input = tf.expand_dims(
        [dataset.tokenizer.word_index['<start>']] * dataset.batch_size, 1)

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


def load_checkpoint(encoder, decoder, optimizer):
    # Checkpoints
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("[*] Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("[*] Initializing from scratch.")

    return ckpt_manager


def train():
    dataset = FlickrDataset()

    vocab_size = len(dataset.tokenizer.word_index) + 1
    num_steps = len(dataset.captions) // dataset.batch_size

    encoder = CNNEncoder(EMBEDDING_DIM)
    decoder = RNNDecoder(EMBEDDING_DIM, UNITS, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt_manager = load_checkpoint(encoder, decoder, optimizer)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    state_save_path = os.path.join(
        'train_states',
        f'loss_plot_{start_epoch - 1}.npy'
    )
    try:
        loss_plot = list(np.load(state_save_path))
    except:
        loss_plot = []

    print('Starting training..')
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset.train_dataset):
            batch_loss, t_loss = train_step(
                img_tensor, target, encoder, decoder, dataset, optimizer)
            total_loss += t_loss

            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        # Save at every epoch
        save_path = ckpt_manager.save()

        print(f'Saved: {save_path}')
        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
        print(f'Time taken for {epoch} epoch {time.time() - start} sec\n')

        state_save_path = os.path.join(
            'train_states',
            f'loss_plot_{epoch}.npy'
        )
        np.save(state_save_path, np.array(loss_plot))


def plot_attention(image, result, attention_plot, title):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    fig.suptitle(title)
    # plt.tight_layout()
    plt.show()


def evaluate(image, encoder, decoder, dataset):
    # dataset = FlickrDataset()
    attention_plot = np.zeros((dataset.max_len, ATTENTION_FEATURE_SHAPE))

    # vocab_size = len(dataset.tokenizer.word_index) + 1

    # optimizer = tf.keras.optimizers.Adam()
    # encoder = CNNEncoder(EMBEDDING_DIM)
    # decoder = RNNDecoder(EMBEDDING_DIM, UNITS, vocab_size)

    hidden = decoder.reset_state(batch_size=1)
    temp_inp = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = dataset.extract_model(temp_inp)
    img_tensor_val = tf.reshape(img_tensor_val,
                                (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_inp = tf.expand_dims([dataset.tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(dataset.max_len):
        predictions, hidden, attention_weights = decoder(
            dec_inp, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(dataset.tokenizer.index_word[predicted_id])

        if dataset.tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_inp = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def test_image(num):
    dataset = FlickrDataset()

    vocab_size = len(dataset.tokenizer.word_index) + 1
    encoder = CNNEncoder(EMBEDDING_DIM)
    decoder = RNNDecoder(EMBEDDING_DIM, UNITS, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt_manager = load_checkpoint(encoder, decoder, optimizer)

    for i in range(num):
        path, titles = dataset.get_random_path()
        result, attention_plot = evaluate(path, encoder, decoder, dataset)

        score = sentence_bleu(titles, result[:-1], weights=(0.5, 0.5))

        if score < 0.7:
            continue

        print('\n')
        print(f'Bleu Score: {score}')
        print(f'Analyzed: {path}')
        print(f'Real: {titles}')
        print(f'Predicted: {result[:-1]}')
        print('\n')
        # plot_attention(path, result, attention_plot, titles)
