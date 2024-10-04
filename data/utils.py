import os
import tensorflow as tf


def create_dir(path):
    exist = os.path.exists(path)
    if not exist:
        os.makedirs(path)
    return exist


def tf_load_image(foldername, timestep):
    file = tf.strings.join([foldername, "/", tf.strings.as_string(timestep), ".gif"])
    return tf.image.decode_gif(tf.io.read_file(file))[0]


def preprocess(img):
    # In range [-1, 1]
    return ((tf.cast(img, tf.float32) / 255.0) * 2.0) - 1.0
