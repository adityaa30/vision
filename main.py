import image_captioning.coco as coco
import image_captioning.vgg16 as vgg16

import keras
import matplotlib.pyplot as plt
import os

# Extract the files
coco.extract_files()

train_ids, train_filenames, train_captions = coco.load_records(train=True)
val_ids, val_filenames, val_captions = coco.load_records(train=False)

print('Total number of training images : {}'.format(len(train_filenames)))
print('Total number of cross-validation images : {}'.format(len(val_filenames)))


def show_image(idx, train):
    """
    Load and plot an image from the training- or validation-set
    with the given index.
    """

    if train:
        # Use an image from the training-set.
        dir = coco.TRAIN_DIR
        filename = train_filenames[idx]
        captions = train_captions[idx]
    else:
        # Use an image from the validation-set.
        dir = coco.VAL_DIR
        filename = val_filenames[idx]
        captions = val_captions[idx]

    # Path for the image-file.
    path = os.path.join(dir, filename)

    # Print the captions for this image.
    x_label = ''
    for caption in captions:
        x_label += caption + '\n'

    # Load the image and show it
    img = coco.load_image(path)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel=x_label)
    plt.title('Filename : {}'.format(filename))
    plt.show()


show_image(idx=25, train=True)

# Process the pre-trained VGG16 model
vgg16_model = keras.applications.vgg16.VGG16(
    weights='imagenet'
)
assert isinstance(vgg16_model, keras.models.Model)

vgg16_model.summary()

layers = vgg16_model.layers
print('Number of layers in VGG16 : {}'.format(len(layers)))

transfer_layer = vgg16_model.get_layer('fc2')
transfer_model = keras.models.Model(
    inputs=vgg16_model.input,
    outputs=transfer_layer.output
)

transfer_values_train = vgg16.process_images(transfer_model, train_filenames, train=True)
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)

transfer_values_train = vgg16.process_images(transfer_model, train_filenames, train=False)
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)
