from neural_network.vgg import vgg_16
import tensorflow as tf
import tensorflow.contrib.slim as slim
from neural_network.network_constants import FEATURE_MAPS
import numpy as np

NUM_BOXES = 5


def ssd(inputs, num_class=1, training=True):
    _, net = vgg_16(inputs, num_classes=0, is_training=training, spatial_squeeze=False)

    inputs = net["vgg_16/conv5/conv5_3"]
    transfer_layers = []
    transfer_layers.append(net["vgg_16/conv4/conv4_3"])

    with tf.variable_scope("orginaly_fc_layers"):
        inputs = slim.conv2d(inputs, 1024, [3, 3], scope="fc6")
        inputs = slim.conv2d(inputs, 1024, [1, 1], scope="fc7")
        transfer_layers.append(inputs)

    with tf.variable_scope("ssd_layers"):
        inputs = slim.conv2d(inputs, 256, [1, 1], 1, activation_fn=tf.nn.relu, padding="SAME", scope="conv8_1")
        inputs = slim.conv2d(inputs, 512, [3, 3], 2, activation_fn=tf.nn.relu, padding="SAME", scope="conv8_2")
        transfer_layers.append(inputs)

        inputs = slim.conv2d(inputs, 128, [1, 1], 1, activation_fn=tf.nn.relu, padding="SAME", scope="conv9_1")
        inputs = slim.conv2d(inputs, 256, [3, 3], 2, activation_fn=tf.nn.relu, padding="SAME", scope="conv9_2")
        transfer_layers.append(inputs)

        inputs = slim.conv2d(inputs, 128, [1, 1], 1, activation_fn=tf.nn.relu, padding="SAME", scope="conv10_1")
        inputs = slim.conv2d(inputs, 256, [3, 3], 2, activation_fn=tf.nn.relu, padding="SAME", scope="conv10_2")
        transfer_layers.append(inputs)

        inputs = slim.conv2d(inputs, 128, [1, 1], 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv11_1")
        inputs = slim.conv2d(inputs, 256, [3, 3], 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv11_2")
        transfer_layers.append(inputs)

    with tf.variable_scope("outputs"):
        # TODO: Add l2 norm before fist output
        # REF: https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad
        # L2 norm: http://mathworld.wolfram.com/L2-Norm.html
        outputs = []
        for tl, fm in zip(transfer_layers, FEATURE_MAPS):
            result = slim.conv2d(tl, (num_class + 4) * len(fm.aspect_ratios), [3, 3],
                                 activation_fn=tf.nn.softsign, padding="SAME")

            result = tf.reshape(result, (-1, fm.width, fm.height, len(fm.aspect_ratios), 5))

            outputs.append(result)

    return outputs


def extract_found_probability(logits):
    return logits[:, :, :, :, 0]


def ssd_true_loc_prediction_loss(labels, predictions):
    with tf.variable_scope("positive_loss"):
        true_mask = extract_found_probability(labels) > 0.5
        true_mask = tf.expand_dims(true_mask, -1)
        true_mask = tf.tile(true_mask, [1, 1, 1, 1, 5])

        total_loss = tf.losses.huber_loss(labels, predictions, true_mask, 0.3,
                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    return total_loss


def ssd_false_loc_prediction_loss(labels, predictions):
    RATIO = 9999

    with tf.variable_scope("negative_loss_layers"):
        false_mask = extract_found_probability(labels) < 0.5
        false_mask = tf.cast(false_mask, tf.float32)
        false_mask = tf.expand_dims(false_mask, -1)
        false_mask = tf.tile(false_mask, [1, 1, 1, 1, 5])

        num_negatives = tf.reduce_sum(extract_found_probability(false_mask))
        num_positives = tf.reduce_sum(extract_found_probability(labels))

        false_predictions_mask = extract_found_probability(predictions * false_mask)
        false_predictions_mask_flat = tf.reshape(false_predictions_mask, (-1,))

        num_tops = tf.minimum(num_negatives, tf.maximum(RATIO * num_positives, RATIO))
        num_tops = tf.cast(num_tops, tf.int32)

        values, _ = tf.math.top_k(false_predictions_mask_flat, num_tops)

        total_loss = tf.cond(tf.less(0, num_tops),
                             lambda: calc_false_loss(num_tops, values, predictions, false_mask, labels), lambda: 0.0)

    return total_loss


def calc_false_loss(num_tops, values, predictions, false_mask, labels):
    minimum_false_with_loss = values[-1]
    loss_false_mask = extract_found_probability(predictions) >= minimum_false_with_loss
    loss_false_mask = tf.cast(loss_false_mask, tf.float32)
    loss_false_mask = loss_false_mask * false_mask[:, :, :, :, 0]

    total_loss = 1/3 * tf.losses.huber_loss(labels[:, :, :, :, 0], predictions[:, :, :, :, 0],
                                            loss_false_mask, delta=0.3,
                                            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return total_loss


def ssd_total_loss(labels, predictions):
    total_loss = 0

    with tf.variable_scope("loss"):
        for label, prediction in zip(labels, predictions):

            total_loss = total_loss \
                         + ssd_true_loc_prediction_loss(label, prediction) \
                         + ssd_false_loc_prediction_loss(label, prediction)

    return total_loss


def simple_loss(labels, predictions):
    return tf.losses.mean_squared_error(labels, predictions)
