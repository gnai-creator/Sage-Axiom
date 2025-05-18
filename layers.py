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
            input_dim=vocab_size, output_dim=dim
        )

    def call(self, x):
        if x.shape.rank != 4:
            raise ValueError(
                f"TokenEmbedding espera input 4D [B, H, W, C]. Recebido: {x.shape}")

        is_onehot = tf.equal(tf.shape(x)[-1], self.vocab_size)
        x = tf.cond(
            is_onehot,
            lambda: tf.argmax(x, axis=-1, output_type=tf.int32),
            lambda: tf.cast(x, tf.int32)
        )

        input_shape = tf.shape(x)
        batch = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]

        flat_x = tf.reshape(x, [-1])
        flat_emb = self.embed_layer(flat_x)

        output = tf.reshape(flat_emb, [batch, H, W, self.dim])
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


class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.buffer = None

    def reset(self):
        self.buffer = None

    def write(self, embedding):
        if self.buffer is None:
            self.buffer = embedding[tf.newaxis, ...]
        else:
            self.buffer = tf.concat(
                [self.buffer, embedding[tf.newaxis, ...]], axis=0)

    def read_all(self):
        if self.buffer is None:
            return tf.zeros((1, 1, 1))
        return self.buffer


class LongTermMemory(tf.keras.layers.Layer):
    def __init__(self, memory_size, embedding_dim):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.memory = self.add_weight(
            shape=(memory_size, embedding_dim),
            initializer='zeros',
            trainable=False,
            name='long_term_memory'
        )

    def store(self, index, embedding):
        update = tf.tensor_scatter_nd_update(
            self.memory, [[index]], [embedding])
        self.memory.assign(update)

    def recall(self, index):
        return tf.expand_dims(tf.gather(self.memory, index), axis=0)


    def match_context(self, context):
        context = tf.reshape(
            context, [tf.shape(context)[0], 1, self.embedding_dim])
        memory = tf.reshape(self.memory, [1, self.memory_size, self.embedding_dim])
        sim = tf.keras.losses.cosine_similarity(context, memory, axis=-1)
        best = tf.argmin(sim, axis=-1)
        return tf.gather(self.memory, best)  # retorna [batch, embedding_dim]



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
        self.rotations = [
            lambda x: x,
            lambda x: tf.image.rot90(x, k=1),
            lambda x: tf.image.rot90(x, k=2),
            lambda x: tf.image.rot90(x, k=3),
        ]
        # A camada Dense é criada aqui, mas o build garante que o input shape esteja certo
        self.selector_layer = tf.keras.layers.Dense(
            4, activation='softmax', name="rotation_selector"
        )

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                "`LearnedRotation` precisa de canais definidos em tempo de compilação. "
                f"Recebido: {input_shape}"
            )
        # Força a construção do selector_layer com input_shape explícito
        self.selector_layer.build((input_shape[0], input_shape[-1]))
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        pooled = tf.reduce_mean(x, axis=[1, 2])  # [batch, channels]
        weights = self.selector_layer(pooled)  # [batch, 4]
        weights = tf.reshape(weights, [b, 4, 1, 1, 1])

        rotated = [rot(x) for rot in self.rotations]
        stacked = tf.stack(rotated, axis=1)  # [batch, 4, h, w, c]
        out = tf.reduce_sum(stacked * weights, axis=1)
        return out


class MeanderHypothesisLayer(tf.keras.layers.Layer):
    def __init__(self, dim, shifts=[(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]):
        super().__init__()
        self.dim = dim
        self.shifts = shifts
        self.conv = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, 1)

    def shift_tensor(self, x, dy, dx):
        return tf.roll(x, shift=[dy, dx], axis=[1, 2])

    def call(self, x):
        base = self.conv(x)
        shifted = [self.shift_tensor(base, dy, dx) for dy, dx in self.shifts]
        shifted.append(base)
        stacked = tf.stack(shifted, axis=0)
        return self.merge(tf.reduce_mean(stacked, axis=0))


class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.base_hypotheses = 4
        self.num_hypotheses = self.base_hypotheses + 1
        self.meander = MeanderHypothesisLayer(dim)
        self.input_proj = tf.keras.layers.Conv2D(
            dim, kernel_size=1, activation='relu')
        self.hypotheses = [
            tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu')
            for _ in range(self.base_hypotheses)
        ]
        self.selector = tf.keras.layers.Dense(
            self.num_hypotheses, activation='softmax')

    def call(self, x, hard=False):
        x = self.input_proj(x)
        candidates = [h(x) for h in self.hypotheses] + [self.meander(x)]

        stacked = tf.stack(candidates, axis=1)

        pooled = tf.reduce_mean(x, axis=[1, 2])
        weights = self.selector(pooled)

        entropy = -tf.reduce_sum(weights *
                                 tf.math.log(weights + 1e-8), axis=-1)
        self.add_loss(0.01 * tf.reduce_mean(entropy))

        if hard:
            idx = tf.argmax(weights, axis=-1)
            one_hot = tf.one_hot(idx, depth=self.num_hypotheses, dtype=tf.float32)[
                :, :, tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        else:
            weights = tf.reshape(weights, [-1, self.num_hypotheses, 1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=1)


class ThresholdMemory(tf.keras.layers.Layer):
    def __init__(self, size=32):
        super().__init__()
        self.size = size
        self.history = self.add_weight(
            shape=(size,), initializer="zeros", trainable=False, name="threshold_history"
        )
        self.pointer = self.add_weight(
            shape=(), initializer="zeros", trainable=False, dtype=tf.int32, name="threshold_ptr"
        )

    def update(self, new_value):
        new_value = tf.reduce_mean(new_value)  # <-- Corrige o shape
        index = self.pointer % self.size
        updated = tf.tensor_scatter_nd_update(
            self.history, [[index]], [new_value])
        self.history.assign(updated)
        self.pointer.assign_add(1)

    def get_adaptive_threshold(self):
        valid = self.history[:tf.minimum(self.pointer, self.size)]
        mean = tf.reduce_mean(valid)
        std = tf.math.reduce_std(valid)
        return mean + tf.random.normal([], stddev=std * 0.5)


class TaskPainSystem(tf.keras.Model):
    def __init__(self, latent_dim=128, task_output_dim=10, **kwargs):
        super(TaskPainSystem, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.task_output_dim = task_output_dim

        # Shared encoder (like a brainstem, if you want to be poetic about it)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=3,
                                   activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3,
                                   activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])

        # Task output (you know, for doing actual work)
        self.task_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(task_output_dim, activation='softmax')
        ])

        # Pain output (when your model hurts)
        self.pain_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='softplus')  # always positive
        ])

    def call(self, pred, expected, blended=None, training=False):

        if blended is None:
            blended = tf.zeros_like(pred)

        x = tf.concat([pred, expected, blended], axis=-1)
        x = self.encoder(x, training=training)
        task_output = self.task_head(x)
        pain_output = self.pain_head(x)
        return {"task": task_output, "pain": pain_output}

    def compute_loss(self, data):
        x, y = data
        y_task = y["task"]
        y_pain = y.get("pain")  # optional

        outputs = self(x, training=True)
        task_loss = self.compiled_loss(y_task, outputs["task"])

        if y_pain is not None:
            pain_loss = keras.losses.mean_squared_error(
                y_pain, outputs["pain"])
            total_loss = task_loss + tf.reduce_mean(pain_loss)
        else:
            total_loss = task_loss

        return total_loss


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


class TaskEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = tf.keras.layers.Conv2D(
            hidden_dim, 3, padding='same', activation='relu')
        self.output_proj = tf.keras.layers.Conv2D(
            hidden_dim, 3, padding='same', activation='relu')
        self.combined_proj = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                hidden_dim, activation='tanh')  # final z_task
        ])

    def call(self, x_in, x_out):
        """
        Recebe dois tensores [B, H, W, 10] (one-hot input/output) e retorna z_task [B, hidden_dim]
        """
        if x_in.shape != x_out.shape:
            raise ValueError("Shape mismatch entre input e output")

        x_concat = tf.concat([
            self.input_proj(x_in),
            self.output_proj(x_out)
        ], axis=-1)

        z = self.combined_proj(x_concat)
        return z  # shape [B, hidden_dim]
