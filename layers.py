# layers.py

import tensorflow.keras as layers
import tensorflow as tf
import logging


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=10, dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim)

    def call(self, x):
        if x.shape.rank != 4:
            raise ValueError(
                f"TokenEmbedding espera input 4D [B, H, W, C]. Recebido: {x.shape}")

        is_onehot = tf.equal(tf.shape(x)[-1], self.vocab_size)

        def _argmax_fn(): return tf.argmax(x, axis=-1, output_type=tf.int32)
        def _cast_fn(): return tf.cast(x, tf.int32)

        x = tf.cond(is_onehot, true_fn=_argmax_fn, false_fn=_cast_fn)

        input_shape = tf.shape(x)
        flat_x = tf.reshape(x, [-1])
        flat_emb = self.embed_layer(flat_x)
        output = tf.reshape(
            flat_emb, [input_shape[0], input_shape[1], input_shape[2], self.dim])
        output.set_shape([None, None, None, self.dim])
        return output


class OutputRefinement(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.refine = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(10, 1)
        ])

    def call(self, x):
        return self.refine(x)


class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        # Corrigir para evitar len(tf.shape(x)) em modo simbólico
        static_rank = x.shape.rank
        while static_rank is not None and static_rank > 4:
            x = tf.reshape(
                x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], -1])
            static_rank = x.shape.rank

        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        y_pos = tf.linspace(-1.0, 1.0, tf.cast(h, tf.int32))
        x_pos = tf.linspace(-1.0, 1.0, tf.cast(w, tf.int32))
        yy, xx = tf.meshgrid(y_pos, x_pos, indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        pos = self.dense(pos)
        return tf.concat([x, pos], axis=-1)


class FractalEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.branch3 = tf.keras.layers.Conv2D(
            dim // 2, kernel_size=3, padding='same', activation='relu')
        self.branch5 = tf.keras.layers.Conv2D(
            dim // 2, kernel_size=5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(
            dim, kernel_size=1, padding='same', activation='relu')
        self.residual = tf.keras.layers.Conv2D(
            dim, kernel_size=1, padding='same')

    def call(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        merged = tf.concat([b3, b5], axis=-1)
        out = self.merge(merged)
        skip = self.residual(x)
        return tf.nn.relu(out + skip)


class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            dim, kernel_size=3, padding='same', activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.skip = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same')

    def call(self, x, training=False):
        out = self.conv(x)
        out = self.bn(out, training=training)
        skip = self.skip(x)
        return tf.nn.relu(out + skip)


class MultiHeadAttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=dim // heads)

    def call(self, x):
        return self.attn(query=x, value=x, key=x)


class LearnedRotation(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.selector_layer = tf.keras.layers.Dense(
            4, activation='softmax', name="rotation_selector")

    def call(self, x):
        def rot_fn(k): return tf.image.rot90(x, k=k)

        b = tf.shape(x)[0]
        pooled = tf.reduce_mean(x, axis=[1, 2])
        weights = self.selector_layer(pooled)
        weights = tf.reshape(weights, [b, 4, 1, 1, 1])

        rotated = [x, rot_fn(1), rot_fn(2), rot_fn(3)]
        stacked = tf.stack(rotated, axis=1)
        out = tf.reduce_sum(stacked * weights, axis=1)
        return out


class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.meander = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim, 1, activation='relu'),
            tf.keras.layers.Conv2D(dim, 1)
        ])
        self.input_proj = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.h1 = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.h2 = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.h3 = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.h4 = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.selector = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, x, hard=False):
        x = self.input_proj(x)
        candidates = [self.h1(x), self.h2(x), self.h3(x),
                      self.h4(x), self.meander(x)]
        stacked = tf.stack(candidates, axis=1)
        pooled = tf.reduce_mean(x, axis=[1, 2])
        weights = self.selector(pooled)
        entropy = -tf.reduce_sum(weights *
                                 tf.math.log(weights + 1e-8), axis=-1)
        self.add_loss(0.01 * tf.reduce_mean(entropy))

        if hard:
            idx = tf.argmax(weights, axis=-1)
            one_hot = tf.one_hot(idx, depth=5, dtype=tf.float32)[
                :, :, tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        else:
            weights = tf.reshape(weights, [-1, 5, 1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=1)

class AttentionOverMemory(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = tf.keras.layers.Dense(dim)
        self.key_proj = tf.keras.layers.Dense(dim)
        self.value_proj = tf.keras.layers.Dense(dim)

    def call(self, memory, query):
        q = self.query_proj(query)[:, tf.newaxis, :]
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        attn_weights = tf.nn.softmax(tf.reduce_sum(
            q * k, axis=-1, keepdims=True), axis=1)
        attended = tf.reduce_sum(attn_weights * v, axis=1)
        return attended


class EnhancedEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.blocks = [
            FractalEncoder(dim),
            FractalBlock(dim),
            FractalBlock(dim),
            FractalBlock(dim),
            tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')
        ]

    def call(self, x, training=False):  # <-- Adiciona training
        x = self.blocks[0](x)
        x = self.blocks[1](x, training=training)
        x = self.blocks[2](x, training=training)
        x = self.blocks[3](x, training=training)
        x = self.blocks[4](x)
        return x


