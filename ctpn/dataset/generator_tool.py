# coding=utf-8

"""
Generator tfrecord file tool class
illool@163.com
"""

import os
import tensorflow as tf
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class GeneratorTfrecordTool(object):
    @staticmethod
    def convert_to_example_simple(image_example, image_buffer):
        """
        covert to tf.train.Example
        :param image_example: dict, an image example
        :param image_buffer: numpy.array, JPEG encoding of RGB image
        :return: Example proto
        """
        image_info = image_example['shape']  # [816 608   3]
        bboxs = np.array(image_example['bboxs'])  # 传入bbox
        bboxs_shape = list(bboxs.shape)  # [21, 5],21个小box,[x_min, y_min, x_max, y_max, 1]1:包含目标;0:不包含目标
        image_encoded = image_buffer.tostring() if isinstance(image_buffer, np.ndarray) else image_buffer
        # example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),  # image
            'image/image_info': tf.train.Feature(int64_list=tf.train.Int64List(value=image_info[0])),  # image_shape
            'image/bboxs_info': tf.train.Feature(int64_list=tf.train.Int64List(value=bboxs_shape)),  # bboxs shape
            'image/bboxs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxs.tostring()]))  # landmark坐标
        }))
        return example

    @staticmethod
    def read_img_and_bboxs(img_path, bbox_path):
        img = cv.imread(img_path)  # img is numpy.array
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        bboxs = GeneratorTfrecordTool.load_annoataion(bbox_path)
        image_example = {"shape": im_info, "bboxs": bboxs}
        return image_example, img

    @staticmethod
    def read_img_and_bboxs_gfile(img_path, bbox_path):
        img = cv.imread(img_path)  # img is numpy.array
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        bboxs = GeneratorTfrecordTool.load_annoataion(bbox_path)
        image_example = {"shape": im_info, "bboxs": bboxs}
        # img = tf.gfile.FastGFile(img_path, 'rb').read()  # img is bytes
        g_img = tf.gfile.GFile(img_path, 'rb').read()  # img is bytes
        return image_example, g_img

    @staticmethod
    def load_annoataion(bbox_path):
        bboxs = []
        with open(bbox_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            x_min, y_min, x_max, y_max = map(int, line)
            bboxs.append([x_min, y_min, x_max, y_max, 1])
        return bboxs

    @staticmethod
    def convert_to_tfrecord_data(data_dir, out_path):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param out_path: string, tfrecord file dir
        :return:
        """
        tf_writer = tf.python_io.TFRecordWriter(out_path)
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(data_dir, "label", '' + bfn + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # 原始文件的方式转换数据
                image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs_gfile(img_path, bboxs_path)
                example = GeneratorTfrecordTool.convert_to_example_simple(image_example, img)
                tf_writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        tf_writer.close()

    @staticmethod
    def convert_to_tfrecord_data_gfile(data_dir, out_path):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param out_path: string, tfrecord file dir
        :return:
        """
        tf_writer = tf.python_io.TFRecordWriter(out_path)
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(data_dir, "label", '' + bfn + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # gfile的方式转换数据
                image_example, img = GeneratorTfrecordTool.read_img_and_bboxs_gfile(img_path, bboxs_path)
                example = GeneratorTfrecordTool.convert_to_example_simple(image_example, img)
                tf_writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        tf_writer.close()

    @staticmethod
    def read_and_decode_tfrecord_data_and_show(tfrecord_name):

        def _parse_record(example_proto):
            features_ = {
                'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
                'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/bboxs': tf.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features=features_)
            return parsed_features

        # 用 dataset 读取 tfrecord 文件g
        dataset = tf.data.TFRecordDataset(tfrecord_name)
        dataset = dataset.map(_parse_record)
        iterator = dataset.make_one_shot_iterator()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        with tf.Session() as sess:
            for i in range(num):
                features = sess.run(iterator.get_next())
                img_data = features['image/encoded']
                shape = features['image/image_info']
                bboxs_shape = features['image/bboxs_info']
                bboxs = features['image/bboxs']
                bboxs = tf.decode_raw(bboxs, tf.int64)
                bboxs = tf.reshape(bboxs, bboxs_shape)
                bboxs_list = bboxs.eval()

                img_data = tf.decode_raw(img_data, tf.uint8)
                # img_data = tf.image.decode_jpeg(img_data)
                img_data = tf.reshape(img_data, shape)
                img_data = tf.cast(img_data, tf.float32) / 255.  # 必须除以255.

                plt.figure()
                # 显示图片
                plt.imshow(img_data.eval())
                for _ in bboxs_list:
                    plt.plot([_[0], _[2], _[2], _[0], _[0]], [_[1], _[1], _[3], _[3], _[1]], 'g', '-.')
                plt.show()
                plt.show()

                # 将数据重新编码成 jpg 图片并保存
                # img = tf.image.encode_jpeg(img_data)
                # tf.gfile.GFile('cat_encode.jpg', 'wb').write(img_data.eval())

    @staticmethod
    def read_and_decode_gfile_tfrecord_data_and_show(tfrecord_name):

        def _parse_record(example_proto):
            features_ = {
                'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
                'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/bboxs': tf.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features=features_)
            return parsed_features

        # 用 dataset 读取 tfrecord 文件g
        dataset = tf.data.TFRecordDataset(tfrecord_name)
        dataset = dataset.map(_parse_record)
        iterator = dataset.make_one_shot_iterator()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        with tf.Session() as sess:
            for i in range(num):
                features = sess.run(iterator.get_next())
                img_data = features['image/encoded']
                shape = features['image/image_info']
                bboxs_shape = features['image/bboxs_info']
                bboxs = features['image/bboxs']
                bboxs = tf.decode_raw(bboxs, tf.int64)
                bboxs = tf.reshape(bboxs, bboxs_shape)
                bboxs_list = bboxs.eval()
                # 恢复为.jpg文件
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)

                img_data = tf.image.decode_jpeg(img_data)
                # img_data = tf.decode_raw(img_data, tf.uint8)
                # image_data = np.reshape(img_data, shape)
                img_data = tf.reshape(img_data, shape)
                img_data = tf.cast(img_data, tf.float32) / 255.  # 必须除以255.

                plt.figure()
                # 显示图片
                plt.imshow(img_data.eval())
                for _ in bboxs_list:
                    plt.plot([_[0], _[2], _[2], _[0], _[0]], [_[1], _[1], _[3], _[3], _[1]], 'r', '-.')
                plt.show()

                # 将数据重新编码成 jpg 图片并保存
                # img_data = features['image/encoded']
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)

    # 该函数用于统计 TFRecord 文件中的样本数量(总数)
    @staticmethod
    def total_sample(tfrecord_name):
        sample_nums = 0
        for record in tf.python_io.tf_record_iterator(tfrecord_name):
            type(record)
            sample_nums += 1
        return sample_nums

    @staticmethod
    def read_gfile_tfrecord_data(tfrecord_name):

        features_ = {
            'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
            'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/bboxs': tf.FixedLenFeature([], tf.string)
        }

        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([tfrecord_name])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features=features_)
        img_data = features['image/encoded']
        shape = features['image/image_info']
        bboxs_shape = features['image/bboxs_info']
        bboxs = features['image/bboxs']
        bboxs = tf.decode_raw(bboxs, tf.int64)
        bboxs = tf.reshape(bboxs, bboxs_shape)
        # bboxs_list = bboxs.eval()
        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.reshape(img_data, shape)
        img_data = tf.expand_dims(img_data, axis=0)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5
        shape = tf.expand_dims(shape, axis=0)
        '''
        print(type(img_data), type(bboxs), type(shape))
        print("img_data:", img_data)
        print("bboxs:", bboxs)
        print("shape:", shape)
        '''
        return img_data, shape, bboxs, bboxs_shape

    @staticmethod
    def read_gfile_tfrecord_data_sess(tfrecord_name):
        img_data, shape, bboxs, bboxs_shape = GeneratorTfrecordTool.read_gfile_tfrecord_data(tfrecord_name)
        # 使用shuffle_batch可以随机打乱输入, capacity就是此队列的容量,min_after_dequeue的值一定要比capacity要小
        # 使用shuffle_batch的前提是图片的shape都都一样,本例下是不能使用的
        # img_data, shape, bboxs, bboxs_shape = tf.train.shuffle_batch([img_data, shape, bboxs, bboxs_shape],
        #                                                              batch_size=1, capacity=2000,
        #                                                              min_after_dequeue=1000)
        init = tf.global_variables_initializer()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            sess.run(init)
            queue_runner = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(num):
                img_data_, shape_, bboxs_, bboxs_shape_ = sess.run([img_data, shape, bboxs, bboxs_shape])
                # do trian :feed_dict = {img_data_:img_data_....}
                print(img_data_.shape, shape_, bboxs_.shape, bboxs_shape_)
                print("img_data:", img_data_)
                print("bboxs:", bboxs_)
                print("shape:", shape_)
            coord.request_stop()
            coord.join(queue_runner)


if __name__ == '__main__':
    GeneratorTfrecordTool.convert_to_tfrecord_data_gfile(
        "/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test",
        "/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")
    nums = GeneratorTfrecordTool.total_sample("/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")
    print(nums)
    GeneratorTfrecordTool.read_and_decode_gfile_tfrecord_data_and_show(
        "/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")

    GeneratorTfrecordTool.convert_to_tfrecord_data(
        "/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test",
        "/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    nums = GeneratorTfrecordTool.total_sample("/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    print(nums)
    GeneratorTfrecordTool.read_and_decode_tfrecord_data_and_show(
        "/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    gpu_id = 1
    print('/gpu:%d' % gpu_id)
    GeneratorTfrecordTool.read_gfile_tfrecord_data_sess("/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")
