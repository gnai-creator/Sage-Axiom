#layers.py
import tensorflow as tf
import logging

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

class DoubtModule(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        pooled = self.global_pool(x)
        h = self.d1(pooled)
        return self.d2(h), h

def compute_auxiliary_loss(output):
    flipped = tf.image.flip_left_right(output)
    symmetry_loss = tf.reduce_mean(tf.square(output - flipped))
    return 0.01 * symmetry_loss

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
        weights = self.selector(tf.reduce_mean(x, axis=[1, 2]))
        if hard:
            idx = tf.argmax(weights, axis=-1)
            one_hot = tf.one_hot(idx, depth=4, dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        else:
            weights = tf.reshape(weights, [-1, 4, 1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=1)

class TaskPainSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.threshold = tf.Variable(1.0, trainable=True)
        self.sensitivity = tf.Variable(tf.ones([1, 1, 1, 10]), trainable=True)
        self.alpha_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self.per_sample_pain = None
        self.adjusted_pain = None
        self.exploration_gate = None
        self.gate = None
        self.alpha = None

        self.confidence = None
        self.entropy = None
        self.ambition = None
        self.assertiveness = None
        self.tenacity = None
        self.faith = None
        self.curiosity = None
        self.patience = None
        self.resilience = None
        self.creativity = None
        self.empathy = None
        self.flexibility = None

    def call(self, pred, expected):
        diff = tf.square(pred - expected)
        self.per_sample_pain = tf.reduce_mean(self.sensitivity * diff, axis=[1, 2, 3], keepdims=True)
        self.exploration_gate = tf.sigmoid((self.per_sample_pain - 5.0) * 0.3)
        self.adjusted_pain = self.per_sample_pain * (1.0 - self.exploration_gate)
        self.gate = tf.sigmoid((self.adjusted_pain - self.threshold) * 2.5)
        self.alpha = self.alpha_layer(self.exploration_gate)

        alpha_loss = 0.01 * tf.reduce_mean(tf.square(self.alpha - 0.5))
        gate_reg_loss = 0.01 * tf.reduce_mean(tf.square(self.exploration_gate - 0.5))
        self.add_loss(alpha_loss)
        self.add_loss(gate_reg_loss)

        logging.info(f"Pain: {self.per_sample_pain.numpy()}, Adjusted: {self.adjusted_pain.numpy()}, Gate: {self.gate.numpy()}, Exploration: {self.exploration_gate.numpy()}, Alpha: {self.alpha.numpy()}")

        return self.adjusted_pain, self.gate, self.exploration_gate, self.alpha

    def compute_trait_loss(self, output_logits, expected):
        probs = tf.nn.softmax(output_logits)
        self.confidence = tf.reduce_mean(tf.reduce_max(probs, axis=-1))
        self.entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1))
        self.ambition = tf.nn.relu(self.exploration_gate - 0.5)
        self.assertiveness = self.gate
        self.tenacity = tf.nn.relu(self.adjusted_pain - 5.0) * (1.0 - self.exploration_gate)
        self.faith = tf.reduce_mean(self.alpha) * self.confidence
        self.curiosity = self.entropy
        self.patience = tf.exp(-self.adjusted_pain)
        self.resilience = tf.exp(-tf.abs(self.per_sample_pain - self.adjusted_pain))
        self.creativity = tf.math.reduce_std(probs)
        self.empathy = tf.reduce_mean(self.alpha) * tf.reduce_mean(self.gate)
        self.flexibility = tf.reduce_mean(tf.abs(output_logits - expected))

        bonus = (
            -0.01 * self.ambition +
            0.01 * self.assertiveness -
            0.01 * self.tenacity -
            0.01 * self.faith +
            0.01 * self.curiosity +
            0.01 * self.patience +
            0.01 * self.resilience +
            0.01 * self.creativity +
            0.01 * self.empathy -
            0.01 * self.flexibility
        )
        entropy_loss = 0.01 * self.entropy
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



class SpectralSynthesizer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.prime_weights = self.add_weight(
            name="prime_weights",
            shape=[dim],
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=False
        )

    def call(self, x):
        primes = tf.constant([2, 3, 5, 7, 11, 13, 17, 19], dtype=tf.float32)
        shifts = [tf.roll(x, shift=int(p), axis=0) for p in primes]
        modulation = tf.add_n([s * tf.math.log(p) for s, p in zip(shifts, primes)])
        return x + modulation

class IdentityCrystallizer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.state = tf.Variable(tf.zeros([dim]), trainable=False)

    def call(self, x):
        update = tf.reduce_mean(x, axis=0)
        self.state.assign(0.9 * self.state + 0.1 * update)
        return self.state

class AffectiveTimeCrystal(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cycle = tf.Variable(tf.zeros([1, dim]), trainable=False)
        self.projector = tf.keras.layers.Dense(dim, activation='tanh')

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        elif len(x.shape) > 2:
            x = tf.reshape(x, [x.shape[0], -1])
        x = tf.stop_gradient(x)
        emotion = self.projector(x)
        self.cycle.assign(0.8 * self.cycle + 0.2 * emotion)
        return self.cycle

class SymbolicContradictionHarvester(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query = tf.keras.layers.Dense(dim)
        self.memory = tf.Variable(tf.zeros([1, dim]), trainable=False)

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        elif len(x.shape) > 2:
            x = tf.reshape(x, [x.shape[0], -1])
        x = tf.stop_gradient(x)
        q = self.query(x)
        self.memory.assign(0.95 * self.memory + 0.05 * q)
        return q - self.memory


class ReflexiveObserver(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.meta = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        elif len(x.shape) > 2:
            x = tf.reshape(x, [x.shape[0], -1])
        return x * self.meta(x)


class HesitationCore(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.uncertainty_proj = tf.keras.layers.Dense(dim, activation='tanh')
        self.conflict_gate = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, contradiction):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        elif len(x.shape) > 2:
            x = tf.reshape(x, [x.shape[0], -1])
        if len(contradiction.shape) == 1:
            contradiction = tf.expand_dims(contradiction, axis=0)
        elif len(contradiction.shape) > 2:
            contradiction = tf.reshape(contradiction, [contradiction.shape[0], -1])
        doubt = self.uncertainty_proj(contradiction)
        gate = self.conflict_gate(doubt)
        hesitant_output = x * (1 - gate) + doubt * gate
        return hesitant_output
