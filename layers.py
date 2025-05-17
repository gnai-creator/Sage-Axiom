#layers.py
import tensorflow as tf
import logging

def compute_auxiliary_loss(output):
    flipped = tf.image.flip_left_right(output)
    symmetry_loss = tf.reduce_mean(tf.square(output - flipped))
    return 0.01 * symmetry_loss

class OutputRefinement(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.refine = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
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
            self.buffer = tf.concat([self.buffer, embedding[tf.newaxis, ...]], axis=0)

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
        update = tf.tensor_scatter_nd_update(self.memory, [[index]], [embedding])
        self.memory.assign(update)

    def recall(self, index):
        return tf.expand_dims(tf.gather(self.memory, index), axis=0)

    def match_context(self, context):
        context = tf.reshape(context, [tf.shape(context)[0], 1, self.embedding_dim])
        memory = tf.reshape(self.memory, [1, self.memory_size, self.embedding_dim])
        sim = tf.keras.losses.cosine_similarity(context, memory, axis=-1)
        best = tf.argmin(sim, axis=-1)
        return self.recall(best)

class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
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
        # Shape: [batch, H, W, C]
        pred_mask = tf.reduce_max(prediction_probs, axis=-1) > self.threshold
        true_mask = tf.reduce_max(expected_onehot, axis=-1) > 0.5

        def safe_bbox(mask, label):
            coords = tf.where(mask)
            count = tf.shape(coords)[0]
            is_empty = tf.equal(count, 0)

            def fallback():
                tf.print(f"⚠️ Bounding box failed for:", label, "- using default values")
                return 0, 0, 1, 1  # min box to avoid divide-by-zero later

            def compute_bbox():
                y_min = tf.reduce_min(coords[:, 0])
                x_min = tf.reduce_min(coords[:, 1])
                y_max = tf.reduce_max(coords[:, 0])
                x_max = tf.reduce_max(coords[:, 1])
                return y_min, x_min, y_max, x_max

            return tf.cond(is_empty, fallback, compute_bbox)

        penalties = []
        batch_size = tf.shape(pred_mask)[0]

        for b in tf.range(batch_size):
            p_y1, p_x1, p_y2, p_x2 = safe_bbox(pred_mask[b], label="pred")
            t_y1, t_x1, t_y2, t_x2 = safe_bbox(true_mask[b], label="true")

            pred_area = tf.cast((p_y2 - p_y1 + 1) * (p_x2 - p_x1 + 1), tf.float32)
            true_area = tf.cast((t_y2 - t_y1 + 1) * (t_x2 - t_x1 + 1), tf.float32)

            area_penalty = tf.nn.relu(pred_area - true_area) / (true_area + 1.0)
            center_offset = tf.sqrt(
                tf.square((p_y1 + p_y2) / 2 - (t_y1 + t_y2) / 2) +
                tf.square((p_x1 + p_x2) / 2 - (t_x1 + t_x2) / 2)
            ) / 20.0

            penalties.append(area_penalty + center_offset)

        return self.penalty_weight * tf.reduce_mean(penalties)

class FractalEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.branch3 = tf.keras.layers.Conv2D(dim // 2, kernel_size=3, padding='same', activation='relu')
        self.branch5 = tf.keras.layers.Conv2D(dim // 2, kernel_size=5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same', activation='relu')
        self.residual = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same')

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
        self.conv = tf.keras.layers.Conv2D(dim, kernel_size=3, padding='same', activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.skip = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same')

    def call(self, x, training=False):  # <-- Adiciona training
        out = self.conv(x)
        out = self.bn(out, training=training)  # <-- Usa o training
        skip = self.skip(x)
        return tf.nn.relu(out + skip)



class MultiHeadAttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)

    def call(self, x):
        return self.attn(query=x, value=x, key=x)

class LearnedRotation(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.rotations = [
            lambda x: x,
            lambda x: tf.image.rot90(x, k=1),
            lambda x: tf.image.rot90(x, k=2),
            lambda x: tf.image.rot90(x, k=3),
        ]
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x):
        b = tf.shape(x)[0]
        pooled = tf.reduce_mean(x, axis=[1,2])  # [B, C]
        weights = self.selector(pooled)         # [B, 4]
        weights = tf.reshape(weights, [b, 4, 1, 1, 1])  # For broadcasting

        rotated = [rot(x) for rot in self.rotations]
        stacked = tf.stack(rotated, axis=1)     # [B, 4, H, W, C]

        out = tf.reduce_sum(stacked * weights, axis=1)
        return out


class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu')
        self.hypotheses = [tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu') for _ in range(4)]
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x, hard=False):
        x = self.input_proj(x)
        candidates = [h(x) for h in self.hypotheses]
        stacked = tf.stack(candidates, axis=1)

        pooled = tf.reduce_mean(x, axis=[1, 2])
        weights = self.selector(pooled)

        # ⛓️ Entropia SEMPRE computada, mesmo se `hard`
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-8), axis=-1)
        self.add_loss(0.01 * tf.reduce_mean(entropy))  # Dá pancadinha educativa no selector

        if hard:
            idx = tf.argmax(weights, axis=-1)
            one_hot = tf.one_hot(idx, depth=4, dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        else:
            weights = tf.reshape(weights, [-1, 4, 1, 1, 1])
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
        updated = tf.tensor_scatter_nd_update(self.history, [[index]], [new_value])
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

        # moved to build for proper graph behavior if needed
        self.sensitivity_init = 0.01
        self.sensitivity_channels = 10

        self.alpha_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.alpha_noise = tf.keras.layers.GaussianNoise(0.05)

        self.doubt_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.doubt_dense1 = tf.keras.layers.Dense(dim, activation='relu', name='dense_9')
        self.doubt_dense2 = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_10')

    def build(self, input_shape):
        # builds sensitivity shape based on last input dim if needed
        self.sensitivity = self.add_weight(
            name="sensitivity",
            shape=(1, 1, 1, self.sensitivity_channels),
            initializer=tf.keras.initializers.Constant(self.sensitivity_init),
            trainable=False
        )

        super().build(input_shape)

    def call(self, pred, expected, blended=None, training=False):
        self.threshold.assign(self.threshold_memory.get_adaptive_threshold())
    
        diff = tf.clip_by_value(tf.square(pred - expected), 0.0, 1.0)
    
        raw_pain = tf.reduce_mean(tf.sqrt(self.sensitivity * diff + 1e-6), axis=[1, 2, 3], keepdims=True)
        raw_pain = tf.clip_by_value(raw_pain, 0.0, 10.0)
    
        mood_mod = 1.0 + 0.01 * tf.sin(raw_pain * 3.14)
        self.per_sample_pain = tf.clip_by_value(raw_pain * mood_mod, 0.0, 10.0)
    
        exploration = tf.sigmoid((self.per_sample_pain - 3.0) * 0.2)
        osc = 1.0 + 0.05 * tf.cos(self.per_sample_pain)
        self.exploration_gate = tf.clip_by_value(exploration * osc, 0.001, 0.98)
    
        safe_denominator = tf.clip_by_value(1.0 + 0.5 * self.exploration_gate, 1e-3, 10.0)
        self.adjusted_pain = tf.clip_by_value(self.per_sample_pain / safe_denominator, 0.0, 10.0)
    
        avg_pain = tf.reduce_mean(self.adjusted_pain)
        self.threshold_memory.update(avg_pain)
    
        try:
            raw_gate = tf.sigmoid((self.adjusted_pain - self.threshold) * 2.5)
            tf.debugging.check_numerics(raw_gate, "NaN in raw_gate")
            self.gate = tf.clip_by_value(raw_gate, 0.0, 1.0)
        except tf.errors.InvalidArgumentError:
            tf.print("⚠️ Fallback: NaN in gate detected. Assigning 0.5")
            self.gate = tf.ones_like(self.adjusted_pain) * 0.5

    
        self.alpha = self.alpha_layer(self.exploration_gate)
        self.alpha = tf.clip_by_value(self.alpha_noise(self.alpha, training=training), 0.001, 0.999)
    
        tf.debugging.check_numerics(self.per_sample_pain, "NaN in per_sample_pain")
        tf.debugging.check_numerics(self.adjusted_pain, "NaN in adjusted_pain")
        tf.debugging.check_numerics(self.gate, "NaN in gate")
        tf.debugging.check_numerics(self.alpha, "NaN in alpha")
    
        self.add_loss(0.01 * tf.reduce_mean(tf.square(self.alpha - 0.5)))
        self.add_loss(0.01 * tf.reduce_mean(tf.square(self.exploration_gate - 0.5)))
    
        if blended is None:
            blended = tf.zeros([tf.shape(pred)[0], 20, 20, self.sensitivity.shape[-1]])
    
        pooled = self.doubt_pool(blended)
        doubt_repr = self.doubt_dense1(pooled)
        doubt_score = self.doubt_dense2(doubt_repr)
        doubt_loss = 0.01 * tf.reduce_mean(tf.square(doubt_repr)) + 0.01 * tf.reduce_mean(doubt_score)
        self.add_loss(doubt_loss)
    
        tf.print("Pain:", self.per_sample_pain,
                 "Adjusted:", self.adjusted_pain,
                 "Gate:", self.gate,
                 "Exploration:", self.exploration_gate,
                 "Alpha:", self.alpha)
    
        return self.adjusted_pain, self.gate, self.exploration_gate, self.alpha



    def compute_trait_loss(self, output_logits, expected):
        probs = tf.nn.softmax(output_logits)
    
        confidence = tf.reduce_mean(tf.reduce_max(probs, axis=-1))
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1))
        ambition = tf.nn.relu(self.exploration_gate - 0.5)
        assertiveness = self.gate
        tenacity = tf.nn.relu(self.adjusted_pain - 5.0) * (1.0 - self.exploration_gate)
        faith = tf.reduce_mean(self.alpha) * confidence
        curiosity = entropy
        patience = tf.exp(-self.adjusted_pain)
        resilience = tf.exp(-tf.abs(self.per_sample_pain - self.adjusted_pain))
        creativity = tf.math.reduce_std(probs)
        empathy = tf.reduce_mean(self.alpha) * tf.reduce_mean(self.gate)
        flexibility = tf.reduce_mean(tf.abs(output_logits - expected))
    
        # Bonus loss baseado nessas "virtudes"
        bonus = (
            -0.01 * ambition +
            0.01 * assertiveness -
            0.01 * tenacity -
            0.01 * faith +
            0.01 * curiosity +
            0.01 * patience +
            0.01 * resilience +
            0.01 * creativity +
            0.01 * empathy -
            0.01 * flexibility
        )
    
        entropy_loss = 0.01 * entropy
        return bonus + entropy_loss



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
        attn_weights = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True), axis=1)
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
