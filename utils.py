# utils.py

import tensorflow as tf

def temporal_symmetry_loss(self, logits_seq):
        flipped = tf.reverse(logits_seq, axis=[1])
        return tf.reduce_mean(tf.square(logits_seq - flipped))

    def spatial_decay_mask(self, shape, decay_rate=0.1):
        h, w = shape[1], shape[2]
        yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
        dist = tf.sqrt(tf.cast(yy**2 + xx**2, tf.float32))
        decay = tf.exp(-decay_rate * dist)
        decay = tf.expand_dims(decay, axis=0)
        decay = tf.expand_dims(decay, axis=-1)
        return decay

    def repetition_penalty(self, logits):
        probs = tf.nn.softmax(logits, axis=-1)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        penalty = tf.reduce_mean(entropy)
        return 0.01 * penalty

    def reverse_penalty(self, logits, expected):
        probs = tf.nn.softmax(logits, axis=-1)
        expected_mask = tf.reduce_max(expected, axis=-1, keepdims=True)
        non_target = 1.0 - expected_mask
        penalty = tf.reduce_mean(probs * non_target)
        return 0.01 * penalty

    def edge_alignment_penalty(self, output_probs):
        # Penaliza se padrões encostarem nas bordas
        edge_mask = tf.ones_like(output_probs[:, :, :, :1])
        edge_mask = tf.concat([
            edge_mask[:, 0:1],  # top edge
            edge_mask[:, -1:],  # bottom edge
            edge_mask[:, :, 0:1],  # left edge
            edge_mask[:, :, -1:]  # right edge
        ], axis=1)
        return 0.01 * tf.reduce_mean(output_probs * edge_mask)
