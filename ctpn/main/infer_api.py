# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from ctpn.nets import model_train as model
from ctpn.rpn_msr.proposal_layer import proposal_layer
from ctpn.text_connector.detectors import TextDetector

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))


tf.app.flags.DEFINE_string(
    # 'test_data_path', '/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test/image', '')
    'test_data_path', '/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/test', '')
tf.app.flags.DEFINE_string('output_path', '/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/Users/sherry/work/pycharm_python/reconstruction_ctpn/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h % 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w % 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(im, checkpoint_path=None):
    with tf.get_default_graph().as_default():
        # 图片
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        # [image_height, image_width, scale_ratios]
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # global_step = tf.Variable(0, trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        pwd = os.getcwd()
        # father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(pwd, r"ctpn/ctpn")
        # print(checkpoint_path)
        # checkpoint_path = father_path

        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.Session() as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('ckpt_state from {}'.format(ckpt_state), 'Restore from {}'.format(model_path))
            print(os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)
            return boxes


if __name__ == '__main__':
    im_ = cv2.imread(r"/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/test/0.png")[:, :, ::-1]
    boxes = main(im_)
    print("res:")
    print(boxes)
