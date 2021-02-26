# coding=utf-8
"""
重构by illool@163.com
qq:122018919
"""
import numpy as np
from crnn.nets.crnn import CRNN
import tensorflow as tf
import datetime


def calculate_accuracy(predicts, labels):
    """
    :param predicts: encoded predict result
    :param labels: ground true label
    :return: accuracy
    """
    assert len(predicts) == len(labels)

    correct_count = 0
    for i, p_label in enumerate(predicts):
        if p_label == labels[i]:
            correct_count += 1

    acc = correct_count / len(predicts)
    return acc, correct_count


def calculate_edit_distance_mean(edit_distences):
    data = np.array(edit_distences)
    data = data[data != 0]
    if len(data) == 0:
        return 0
    return np.mean(data)


def process_img_back(image_value, img_channels=3):
    img_decoded = tf.image.decode_image(image_value, channels=img_channels)
    if img_channels == 3:
        img_decoded = tf.image.rgb_to_grayscale(img_decoded)

    img_decoded = tf.cast(img_decoded, tf.float32)
    img_decoded = (img_decoded - 128.0) / 128.0

    return img_decoded


def process_img(image_value, img_channels=3):
    # img_decoded = tf.image.decode_image(image_value, channels=img_channels)
    if img_channels == 3:
        img_decoded = tf.image.rgb_to_grayscale(image_value)
    else:
        img_decoded = image_value

    img_decoded = tf.cast(img_decoded, tf.float32)
    img_decoded = (img_decoded - 128.0) / 128.0

    return img_decoded


def process_img_list(image_value_list, img_channels=3):
    # img_decoded = tf.image.decode_image(image_value, channels=img_channels)
    new_list = []
    for image_value in image_value_list:
        if img_channels == 3:
            img_decoded = tf.image.rgb_to_grayscale(image_value)
        else:
            img_decoded = image_value

        img_decoded = tf.cast(img_decoded, tf.float32)
        img_decoded = (img_decoded - 128.0) / 128.0
        new_list.append(img_decoded)

    return new_list


def _sparse_tuple_from_label(sequences, default_val=-1, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
                  encode label, e.g: [2,44,11,55]
        default_val: value should be ignored in sequences
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        seq_filtered = list(filter(lambda x: x != default_val, seq))
        indices.extend(zip([n] * len(seq_filtered), range(len(seq_filtered))))
        values.extend(seq_filtered)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)

    if len(indices) == 0:
        shape = np.asarray([len(sequences), 0], dtype=np.int64)
    else:
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def validation_api(sess, feeds, fetches, img_raw, converter, name):
    """
    :param sess: tensorflow session
    :return: predicts
    """
    predicts = []

    img_batch_ = process_img(img_raw)

    img_batch = sess.run(img_batch_)
    sparse_label_batch = _sparse_tuple_from_label([[0]])

    feed = {feeds['inputs']: [img_batch],
            feeds['labels']: sparse_label_batch,
            feeds['is_training']: False}
    now_time = datetime.datetime.now()
    batch_predicts, edit_distance, batch_edit_distances = sess.run(fetches, feed)
    print(datetime.datetime.now() - now_time)
    batch_predicts = [converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in batch_predicts]
    print(batch_predicts)
    predicts.extend(batch_predicts)
    return predicts


if __name__ == '__main__':
    indices_, values_, shape_ = _sparse_tuple_from_label([[0]])
    print(indices_, values_, shape_)
