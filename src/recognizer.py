import logging

from keras.layers import StringLookup
from keras.saving.save import load_model
from tensorflow import keras

import tensorflow as tf
import numpy as np
import os

# Recognition constants
BATCH_SIZE = 64
PADDING_TOKEN = 99
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
CACHED_MODEL_PATH = 'resources/cache/trained.h5'

# Global variables
char_to_num_converter = None
num_to_char_converter = None
max_label_length = 0

logger = logging.getLogger('recognizer')


def _split_dataset(base_dir):
    words_list = []

    words = open(f"{base_dir}/words.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            words_list.append(line)

    np.random.shuffle(words_list)

    """split the dataset into three subsets with a 90:5:5 ratio (train:validation:test)."""

    split_idx = int(0.9 * len(words_list))
    train_samples = words_list[:split_idx]
    test_samples = words_list[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    assert len(words_list) == len(train_samples) + len(validation_samples) + len(
        test_samples
    )

    return train_samples, validation_samples, test_samples


def get_image_paths_and_labels(base_dir, samples):
    base_image_path = os.path.join(base_dir, "words")
    paths = []
    corrected_samples = []

    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


def _clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)

    return cleaned_labels


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def decode_batch_predictions(pred, num_to_char, max_len):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def preprocess_image(image_path, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_model(char_to_num):
    # Inputs to the model
    input_img = keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((IMAGE_WIDTH // 4), (IMAGE_HEIGHT // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)

    # Print built model details
    model.summary()

    return model


def calculate_edit_distance(labels, max_len, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model, max_len, validation_images, validation_labels):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, self.max_len, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


def decode_batch_predictions(pred, max_len, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def vectorize_label(label):
    label = char_to_num_converter(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_label_length - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=PADDING_TOKEN)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)


def get_trained_model():
    global char_to_num_converter
    global num_to_char_converter
    global max_label_length

    base_dir = 'resources/datasets/IAM_Words'

    train_samples, validation_samples, test_samples = _split_dataset(base_dir)

    logger.debug(f"Total training samples: {len(train_samples)}")
    logger.debug(f"Total validation samples: {len(validation_samples)}")
    logger.debug(f"Total test samples: {len(test_samples)}")

    # Prepare image paths
    train_img_paths, train_labels = get_image_paths_and_labels(base_dir, train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(base_dir, validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(base_dir, test_samples)

    # Find maximum length and the size of the vocabulary in the training data.
    train_labels_cleaned = []
    characters = set()
    max_label_length = 0

    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)

        max_label_length = max(max_label_length, len(label))
        train_labels_cleaned.append(label)

    characters = sorted(list(characters))

    logger.debug("Maximum length: ", max_label_length)
    logger.debug("Vocab size: ", len(characters))

    # Now we clean the validation and the test labels as well
    validation_labels_cleaned = _clean_labels(validation_labels)
    test_labels_cleaned = _clean_labels(test_labels)

    # Mapping characters to integers.
    char_to_num_converter = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char_converter = StringLookup(
        vocabulary=char_to_num_converter.get_vocabulary(), mask_token=None, invert=True
    )

    if os.path.exists(CACHED_MODEL_PATH):
        model = load_model(CACHED_MODEL_PATH, custom_objects={'CTCLayer': CTCLayer})
        return keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)

    """## Prepare `tf.data.Dataset` objects"""

    train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)

    validation_images = []
    validation_labels = []

    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    # Create model and train it
    model = build_model(char_to_num_converter)

    prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
    edit_distance_callback = EditDistanceCallback(
        prediction_model,
        max_label_length,
        validation_images,
        validation_labels,
    )

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=1,
        callbacks=[edit_distance_callback],
    )

    model.save(CACHED_MODEL_PATH)

    return prediction_model


def predict_single_object(path, model):
    image = list(prepare_dataset([path], ['']))[0]['image']
    predictions = model.predict(image)

    return decode_batch_predictions(predictions, max_label_length, num_to_char_converter)
