# coding=utf-8
"""
重构by illool@163.com
qq:122018919
"""
import os
import tensorflow as tf
from crnn.libs.infer_api import validation_api
from crnn.libs.label_converter import LabelConverter
import cv2 as cv
from crnn.parse_args import parse_args
import datetime


# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
def restore_ckpt(sess, checkpoint_dir):
    print("Restoring checkpoint from: " + checkpoint_dir)

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    print("ckpt:", ckpt)
    if ckpt is None:
        print("Checkpoint not found")
        exit(-1)

    meta_file = ckpt + '.meta'
    try:
        print('Restore graph from {}'.format(meta_file))
        print('Restore variables from {}'.format(ckpt))
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, ckpt)
    except Exception:
        raise Exception("Can not restore from {}".format(checkpoint_dir))


def infer(img):
    pwd = os.getcwd()
    chn_text = os.path.join(pwd, r"crnn/data/chars/chn.txt")
    checkpoint_path = os.path.join(pwd, r"crnn/output/checkpoint/default")
    # father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    converter = LabelConverter(chars_file=chn_text)
    crnn_graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config, graph=crnn_graph)
    with crnn_graph.as_default():
        image_value = tf.convert_to_tensor(img, tf.uint8)
        restore_ckpt(sess, checkpoint_path)

        # for node in sess.graph.as_graph_def().node:
        #     print(node.name)

        # https://stackoverflow.com/questions/46912721/tensorflow-restore-model-with-sparse-placeholder
        labels_placeholder = tf.SparseTensor(
            values=sess.graph.get_tensor_by_name('labels/values:0'),
            indices=sess.graph.get_tensor_by_name('labels/indices:0'),
            dense_shape=sess.graph.get_tensor_by_name('labels/shape:0')
        )

        feeds = {
            'inputs': sess.graph.get_tensor_by_name('inputs:0'),
            'is_training': sess.graph.get_tensor_by_name('is_training:0'),
            'labels': labels_placeholder
        }

        fetches = [
            sess.graph.get_tensor_by_name('SparseToDense:0'),  # dense_decoded
            sess.graph.get_tensor_by_name('Mean_1:0'),  # mean edit distance
            sess.graph.get_tensor_by_name('edit_distance:0')  # batch edit distances
        ]

        res = validation_api(sess, feeds, fetches, image_value, converter, name='infer')
        return res


def infer_list(img_list):
    pwd = os.getcwd()
    chn_text = os.path.join(pwd, r"crnn/data/chars/chn.txt")
    checkpoint_path = os.path.join(pwd, r"crnn/output/checkpoint/default")
    # father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # image_value = tf.read_file('./data/demo/00000015.jpg')
    '''
    image_value_list = []
    for img in img_list:
        b, g, r = cv.split(img)
        img = cv.merge([r, g, b])
        image_value = tf.convert_to_tensor(img, tf.uint8)
        image_value_list.append(image_value)
    '''
    converter = LabelConverter(chars_file=chn_text)
    crnn_graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config, graph=crnn_graph)
    with crnn_graph.as_default():
        restore_ckpt(sess, checkpoint_path)

        # for node in sess.graph.as_graph_def().node:
        #     print(node.name)

        # https://stackoverflow.com/questions/46912721/tensorflow-restore-model-with-sparse-placeholder
        labels_placeholder = tf.SparseTensor(
            values=sess.graph.get_tensor_by_name('labels/values:0'),
            indices=sess.graph.get_tensor_by_name('labels/indices:0'),
            dense_shape=sess.graph.get_tensor_by_name('labels/shape:0')
        )

        feeds = {
            'inputs': sess.graph.get_tensor_by_name('inputs:0'),
            'is_training': sess.graph.get_tensor_by_name('is_training:0'),
            'labels': labels_placeholder
        }

        fetches = [
            sess.graph.get_tensor_by_name('SparseToDense:0'),  # dense_decoded
            sess.graph.get_tensor_by_name('Mean_1:0'),  # mean edit distance
            sess.graph.get_tensor_by_name('edit_distance:0')  # batch edit distances
        ]
        res = []
        for img in img_list:
            now_time = datetime.datetime.now()
            b, g, r = cv.split(img)
            img = cv.merge([r, g, b])
            image_value = tf.convert_to_tensor(img, tf.uint8)
            res_t = validation_api(sess, feeds, fetches, image_value, converter, name='infer')
            print(datetime.datetime.now()-now_time)
            res += res_t
        return res


def main():
    args = parse_args(infer=True)
    if args.gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
    print("dev:", dev)
    img = cv.imread(r'./data/demo/00000014.jpg')
    with tf.device(dev):
        res_list = infer(img)
        # print(res_list)
    return res_list


def main_api(img):
    args = parse_args(infer=True)
    if args.gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
    print("dev:", dev)
    with tf.device(dev):
        res_list = infer(img)
        # print(res_list)
    return res_list


def main_api_list(img_list):
    args = parse_args(infer=True)
    if args.gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
    print("dev:", dev)
    with tf.device(dev):
        res_list = infer_list(img_list)
        # print(res_list)
    return res_list


if __name__ == '__main__':
    predicts = main()
    print(predicts)
