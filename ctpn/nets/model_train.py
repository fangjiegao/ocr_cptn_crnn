import tensorflow as tf
from tensorflow.contrib import slim

from ctpn.nets import vgg
from ctpn.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]  # 获得多少个通道
    if len(means) != num_channels:  # 通道数是不是匹配
        raise ValueError('len(means) must match the number of channels')
    # 将axis=3,第四维num_channels分成num_channels份,每份shape=[images[0],images[1],images[2],1]
    # 3通道就是分成3份，每份代表一个通道，下面每个通道进行归一化
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)  # 切片:则value沿维度axis分割成为num_split
    for i in range(num_channels):
        channels[i] -= means[i]  # 遍历每个通道,归一化
    return tf.concat(axis=3, values=channels)  # 再合并成三通道


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C])
        net.set_shape([None, None, input_channel])  # batch_szie, max_time, depth=input_channel=512

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def model(image):
    image = mean_image_subtraction(image)  # 图像归一化
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)  # 得到vgg16的特征  N*H*W*C vgg16中C=521
    # 512,[卷积核个数](输出通道数) kernel_size,[卷积核的高度，卷积核的宽度]
    # rpn_conv = slim.conv2d(conv5_3, 512, [3,3])
    rpn_conv = slim.conv2d(conv5_3, 512, 3)  # (N, H, W, 512)
    # 双向lstm网络net, input_channel=512, hidden_unit_num=128, output_channel=512, scope_name
    lstm_output = Bilstm(rpn_conv, 512, 128, 512, scope_name='BiLSTM')  # N, H, W, 512
    # 10个anchors,每个anchor有4个坐标点(xmin，xmax，ymin，ymax)
    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")
    # 每个anchor有2个类别，是文本or不是文本
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d) 1×H×W×(10x4)
    # 分类
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
                          name="cls_prob")

    return bbox_pred, cls_pred, cls_prob


# cls_pred:分类器,bbox真实的bbox,im_info:图片的shape
def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        # rpn_bbox_targets: 和gt真值的偏差
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info, [16, ], [16]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    # rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        # tf.less 返回 True or False; a<b,返回True， 否则返回False。
        # 0.5*x^2 if abs(x)<1
        # abs(x)-0.5 else
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(bbox_pred, cls_pred, bbox, im_info):
    # rpn_data = (rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])  # rpn_data[0]: rpn_labels
    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))  # -1样本不参加训练
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
    rpn_label = tf.gather(rpn_label, rpn_keep)
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

    # box loss
    rpn_bbox_pred = bbox_pred  # 预测值
    rpn_bbox_targets = rpn_data[1]  # rpn_bbox_targets,和真值之间的偏差
    rpn_bbox_inside_weights = rpn_data[2]  # rpn_bbox_inside_weights
    rpn_bbox_outside_weights = rpn_data[3]  # rpn_bbox_outside_weights

    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)  # -1样本不参加训练
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)  # -1样本不参加训练
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)  # -1样本不参加训练

    rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])
    # sum 所有真样本
    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

    model_loss = rpn_cross_entropy + rpn_loss_box

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
