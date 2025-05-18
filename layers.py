# layers.py

import tensorflow as tf
import logging


def compute_auxiliary_loss(output):
    flipped = tf.image.flip_left_right(output)
    symmetry_loss = tf.reduce_mean(tf.square(output - flipped))
    return 0.01 * symmetry_loss


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=10, dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim)

    def call(self, x):
        # x: [..., 10] → argmax → [...], embedding → [..., dim]
        if x.shape.rank == 4 and x.shape[-1] == self.vocab_size:
            x = tf.argmax(x, axis=-1)
        return self.embed_layer(x)


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
        memory = tf.reshape(
            self.memory, [1, self.memory_size, self.embedding_dim])
        sim = tf.keras.losses.cosine_similarity(context, memory, axis=-1)
        best = tf.argmin(sim, axis=-1)
        return self.recall(best)


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


class BoundingBoxDiscipline(tf.keras.layers.Layer):
    def __init__(self, threshold=0.3, penalty_weight=0.05):
        super().__init__()
        self.threshold = threshold
        self.penalty_weight = penalty_weight

    def call(self, prediction_probs, expected_onehot):
        pred_mask = tf.reduce_max(prediction_probs, axis=-1) > self.threshold
        true_mask = tf.reduce_max(expected_onehot, axis=-1) > 0.5

        def compute_penalty(pair):
            pred, true = pair

            def safe_bbox(mask):
                coords = tf.where(mask)
                count = tf.shape(coords)[0]
                is_empty = tf.equal(count, 0)

                def fallback():
                    return tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)

                def compute():
                    y_min = tf.cast(tf.reduce_min(coords[:, 0]), tf.float32)
                    x_min = tf.cast(tf.reduce_min(coords[:, 1]), tf.float32)
                    y_max = tf.cast(tf.reduce_max(coords[:, 0]), tf.float32)
                    x_max = tf.cast(tf.reduce_max(coords[:, 1]), tf.float32)
                    return tf.stack([y_min, x_min, y_max, x_max])

                return tf.cond(is_empty, fallback, compute)

            p_box = safe_bbox(pred)
            t_box = safe_bbox(true)

            pred_area = (p_box[2] - p_box[0] + 1.0) * \
                (p_box[3] - p_box[1] + 1.0)
            true_area = (t_box[2] - t_box[0] + 1.0) * \
                (t_box[3] - t_box[1] + 1.0)

            area_penalty = tf.nn.relu(
                pred_area - true_area) / (true_area + 1.0)

            center_offset = tf.sqrt(
                tf.square((p_box[0] + p_box[2]) / 2.0 - (t_box[0] + t_box[2]) / 2.0) +
                tf.square((p_box[1] + p_box[3]) / 2.0 -
                          (t_box[1] + t_box[3]) / 2.0)
            ) / 20.0

            return area_penalty + center_offset

        penalties = tf.map_fn(
            compute_penalty, (pred_mask, true_mask), fn_output_signature=tf.float32)

        return self.penalty_weight * tf.reduce_mean(penalties)


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
        self.selector = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "`LearnedRotation` precisa de canais definidos em tempo de compilação. "
                f"Recebido: {input_shape}"
            )

        self.selector = tf.keras.layers.Dense(
            4,
            activation='softmax',
            name="rotation_selector"
        )
        # Construir explicitamente
        self.selector.build((None, channels))
        super().build(input_shape)

    def call(self, x):
        if not self.built:
            self.build(x.shape)

        b = tf.shape(x)[0]
        pooled = tf.reduce_mean(x, axis=[1, 2])  # [batch, channels]
        weights = self.selector(pooled)  # [batch, 4]
        weights = tf.reshape(weights, [b, 4, 1, 1, 1])

        rotated = [rot(x) for rot in self.rotations]
        stacked = tf.stack(rotated, axis=1)
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


class TaskPainSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.threshold_memory = ThresholdMemory()
        self.threshold = tf.Variable(0.05, trainable=False)

        self.sensitivity_init = 0.01
        self.sensitivity_channels = 10

        self.alpha_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.alpha_noise = tf.keras.layers.GaussianNoise(0.05)

        self.doubt_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.doubt_dense1 = tf.keras.layers.Dense(
            dim, activation='relu', name='dense_9')
        self.doubt_dense2 = tf.keras.layers.Dense(
            1, activation='sigmoid', name='dense_10')

    def build(self, input_shape):
        self.sensitivity = self.add_weight(
            name="sensitivity",
            shape=(1, 1, 1, self.sensitivity_channels),
            initializer=tf.keras.initializers.Constant(self.sensitivity_init),
            trainable=False
        )
        super().build(input_shape)

    def call(self, pred, expected, blended=None, training=False):
        threshold_val = self.threshold_memory.get_adaptive_threshold()
        threshold_val = tf.cond(
            tf.math.logical_or(tf.math.is_nan(threshold_val),
                               tf.math.is_inf(threshold_val)),
            lambda: tf.constant(0.05, dtype=tf.float32),
            lambda: threshold_val
        )

        self.threshold.assign(threshold_val)

        diff = tf.clip_by_value(tf.square(pred - expected), 0.0, 1.0)
        raw_pain = tf.reduce_mean(
            tf.sqrt(self.sensitivity * diff + 1e-6), axis=[1, 2, 3], keepdims=True)
        raw_pain = tf.clip_by_value(raw_pain, 0.0, 10.0)

        mood_mod = 1.0 + 0.01 * tf.sin(raw_pain * 3.14)
        per_sample_pain = tf.clip_by_value(
            raw_pain * (1.0 + 0.05 * tf.sin(raw_pain * 6.28)) * mood_mod, 0.0, 10.0)

        exploration = tf.clip_by_value(
            tf.pow(per_sample_pain + 1e-3, 0.65) *
            (1.0 + 0.02 * tf.sin(6.28 * per_sample_pain)),
            0.3, 0.99
        )

        osc = 1.0 + 0.05 * tf.cos(per_sample_pain)
        exploration_gate = tf.clip_by_value(exploration * osc, 0.001, 0.98)

        safe_denominator = tf.clip_by_value(
            1.0 + 0.5 * exploration_gate, 1e-3, 10.0)
        adjusted_pain = tf.clip_by_value(
            per_sample_pain / safe_denominator, 0.0, 10.0)

        avg_pain = tf.reduce_mean(adjusted_pain)
        self.threshold_memory.update(avg_pain)

        raw_gate_input = (adjusted_pain - self.threshold) * 2.5
        raw_gate_input = tf.clip_by_value(raw_gate_input, -20.0, 20.0)
        raw_gate = tf.sigmoid(raw_gate_input)
        tf.debugging.check_numerics(raw_gate, "NaN in raw_gate")
        gate = tf.clip_by_value(raw_gate, 0.0, 1.0)

        alpha = self.alpha_layer(exploration_gate)
        alpha = tf.clip_by_value(self.alpha_noise(
            alpha, training=training), 0.001, 0.8)

        # Safe checks
        tf.debugging.check_numerics(per_sample_pain, "NaN in per_sample_pain")
        tf.debugging.check_numerics(adjusted_pain, "NaN in adjusted_pain")
        tf.debugging.check_numerics(gate, "NaN in gate")
        tf.debugging.check_numerics(alpha, "NaN in alpha")

        # Expose for external use
        self.per_sample_pain = per_sample_pain
        self.adjusted_pain = adjusted_pain
        self.exploration_gate = exploration_gate
        self.gate = gate
        self.alpha = alpha

        self.add_loss(0.01 * tf.reduce_mean(tf.square(alpha - 0.5)))
        self.add_loss(0.01 * tf.reduce_mean(tf.square(exploration_gate - 0.5)))

        if blended is None:
            blended = tf.zeros([
                tf.shape(pred)[0],
                tf.shape(pred)[1],
                tf.shape(pred)[2],
                self.sensitivity.shape[-1]
            ])

        pooled = self.doubt_pool(blended)
        doubt_repr = self.doubt_dense1(pooled)
        doubt_score = self.doubt_dense2(doubt_repr)
        doubt_loss = 0.01 * \
            tf.reduce_mean(tf.square(doubt_repr)) + 0.01 * \
            tf.reduce_mean(doubt_score)
        self.add_loss(doubt_loss)

        # tf.print("Pain:", per_sample_pain,
        #          "Adjusted:", adjusted_pain,
        #          "Gate:", gate,
        #          "Exploration:", exploration_gate,
        #          "Alpha:", alpha)

        return adjusted_pain, gate, exploration_gate, alpha

    def compute_trait_loss(self, output_logits, expected, per_sample_pain):
        probs = tf.nn.softmax(output_logits)

        confidence = tf.reduce_mean(tf.reduce_max(probs, axis=-1))
        entropy = - \
            tf.reduce_mean(tf.reduce_sum(
                probs * tf.math.log(probs + 1e-8), axis=-1))
        entropy = tf.clip_by_value(entropy, 0.0, 2.3)
        # quanto menor a entropia, maior o impacto
        entropy_scale = tf.clip_by_value(1.0 - entropy, 0.1, 1.0)
        ambition = tf.nn.relu(self.exploration_gate - 0.5)
        assertiveness = self.gate
        tenacity = tf.nn.relu(self.adjusted_pain - 5.0) * \
            (1.0 - self.exploration_gate)
        faith = tf.reduce_mean(self.alpha) * confidence
        curiosity = entropy
        patience = tf.exp(-self.adjusted_pain)
        resilience = tf.exp(-tf.abs(per_sample_pain - self.adjusted_pain))
        creativity = tf.math.reduce_std(probs)
        empathy = tf.reduce_mean(self.alpha) * tf.reduce_mean(self.gate)
        flexibility = tf.reduce_mean(
            tf.abs(tf.nn.softmax(output_logits) - expected))
        flexibility = tf.clip_by_value(flexibility, 0.0, 1.0)
        # penaliza confiança muito longe de 0.5
        confidence_penalty = tf.reduce_mean(tf.square(confidence - 0.5)) * 0.1
        logit_std = tf.math.reduce_std(output_logits)
        logit_penalty = tf.clip_by_value(logit_std - 2.0, 0.0, 10.0) * 0.01
        # bonus = (
        #     +0.02 * curiosity        # incentivo principal: explorar diferentes hipóteses
        #     +0.01 * resilience       # não quebrar ao errar
        #     +0.01 * patience         # evita decisões impulsivas
        #     +0.01 * creativity       # variedade ajuda em tarefas abertas
        #     +0.005 * empathy         # ponderar riscos com confiança
        #     +0.005 * flexibility     # mudar quando necessário
        #     -0.005 * tenacity        # penaliza insistência excessiva
        #     -0.005 * faith           # penaliza crença exagerada em outputs imprecisos
        #     -0.005 * assertiveness   # controla decisões super decisivas sem base
        # )

        bonus = (
            0.01 * ambition +
            0.01 * assertiveness +
            0.01 * tenacity +
            0.01 * faith +
            0.02 * curiosity +
            0.01 * patience +
            0.01 * resilience +
            0.01 * creativity +
            0.01 * empathy +
            0.01 * flexibility
        )
        bonus *= entropy_scale
        bonus = tf.clip_by_value(bonus, -0.2, 0.2)
        entropy_loss = 0.01 * entropy
        total_loss = bonus + entropy_loss + confidence_penalty + logit_penalty
        return tf.clip_by_value(total_loss, 0.0, 1.0)


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
