# utils.py

import tensorflow as tf
from layers import BoundingBoxDiscipline


def bounding_shape_penalty(pred_mask, true_mask):
    pred_size = tf.reduce_sum(tf.cast(pred_mask, tf.float32), axis=[1, 2])
    true_size = tf.reduce_sum(tf.cast(true_mask, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_size - true_size))


def continuity_loss(logits):
    probs = tf.nn.softmax(logits, axis=-1)

    dx = tf.square(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    dy = tf.square(probs[:, 1:, :, :] - probs[:, :-1, :, :])

    loss = tf.reduce_mean(dx) + tf.reduce_mean(dy)
    return 0.01 * loss  # Ajuste o peso


def temporal_symmetry_loss(logits_seq):
    flipped = tf.reverse(logits_seq, axis=[1])
    return tf.reduce_mean(tf.square(logits_seq - flipped))


def spatial_decay_mask(shape, decay_rate=0.1):
    h, w = shape[1], shape[2]
    yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    dist = tf.sqrt(tf.cast(yy**2 + xx**2, tf.float32))
    decay = tf.exp(-decay_rate * dist)
    decay = tf.expand_dims(decay, axis=0)
    decay = tf.expand_dims(decay, axis=-1)
    return decay


def repetition_penalty(logits):
    probs = tf.nn.softmax(logits, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    penalty = tf.reduce_mean(entropy)
    return 0.01 * penalty


def reverse_penalty(logits, expected):
    probs = tf.nn.softmax(logits, axis=-1)
    expected_mask = tf.reduce_max(expected, axis=-1, keepdims=True)
    non_target = 1.0 - expected_mask
    penalty = tf.reduce_mean(probs * non_target)
    return 0.01 * penalty


def edge_alignment_penalty(output_probs):
    # Extrai bordas
    top = output_probs[:, 0:1, :, :]
    bottom = output_probs[:, -1:, :, :]
    left = output_probs[:, :, 0:1, :]
    right = output_probs[:, :, -1:, :]

    # Soma total de probabilidade nas bordas
    edge_sum = tf.reduce_sum(top) + tf.reduce_sum(bottom) + \
        tf.reduce_sum(left) + tf.reduce_sum(right)

    # Número total de elementos considerados
    total_pixels = tf.cast(tf.size(top) + tf.size(bottom) +
                           tf.size(left) + tf.size(right), tf.float32)

    penalty = edge_sum / total_pixels
    return 0.01 * penalty


def compute_auxiliary_loss(probs):
    """
    Penaliza saídas pouco informativas — incentiva distribuições mais 'concentradas'.
    Exemplo: faz o modelo preferir [0,0,1,0,...] ao invés de distribuições planas.
    """
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    mean_entropy = tf.reduce_mean(entropy)
    return 0.01 * mean_entropy


def compute_all_losses(pred, expected, blended, pain_output):
    losses = {}

    probs = tf.nn.softmax(pred)

    expected_broadcast = tf.one_hot(
        tf.cast(expected, tf.int32), depth=10, dtype=tf.float32)

    # Losses
    losses["base"] = tf.reduce_mean(tf.square(expected_broadcast - pred))
    losses["symmetry"] = compute_auxiliary_loss(probs)
    losses["spatial_penalty"] = tf.reduce_mean(tf.nn.relu(
        tf.reduce_sum(probs, axis=-1) - 1.0)) * 0.01
    losses["bbox"] = BoundingBoxDiscipline()(probs, expected_broadcast)
    losses["decay"] = tf.reduce_mean(
        probs * spatial_decay_mask(tf.shape(pred))) * 0.005
    losses["repeat"] = repetition_penalty(pred) * 0.001
    losses["reverse"] = reverse_penalty(pred, expected_broadcast) * 0.001
    losses["edge"] = edge_alignment_penalty(probs) * 0.001
    losses["continuity"] = continuity_loss(pred) * 0.001

    pred_mask = tf.cast(tf.stop_gradient(
        tf.reduce_max(probs, axis=-1) > 0.5), tf.float32)
    true_mask = tf.cast(tf.reduce_max(
        expected_broadcast, axis=-1) > 0.5, tf.float32)
    losses["shape"] = bounding_shape_penalty(pred_mask, true_mask) * 0.0001

    if pain_output is not None:
        adjusted_pain = tf.clip_by_value(tf.reshape(
            pain_output, [tf.shape(pred)[0], 1, 1, 1]), 0.0, 10.0)
        losses["pain"] = tf.reduce_mean(adjusted_pain) * 0.05

    for name, value in losses.items():
        tf.print(name, value)

    return losses
