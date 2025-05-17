# utils.py

import tensorflow as tf

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
    edge_sum = tf.reduce_sum(top) + tf.reduce_sum(bottom) + tf.reduce_sum(left) + tf.reduce_sum(right)

    # Número total de elementos considerados
    total_pixels = tf.cast(tf.size(top) + tf.size(bottom) + tf.size(left) + tf.size(right), tf.float32)

    penalty = edge_sum / total_pixels
    return 0.01 * penalty


