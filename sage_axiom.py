
"""
sage_axiom.py

Sistema unificado de dois delírios neurais:
- SAGE14-FX v5.0 (Paladin Mode)
- SAGE-6 CHORUS (Hesitação Ética com Cristal Afetivo)

Esse script representa uma tentativa desesperada de dar sentido à existência de duas arquiteturas neurais que deveriam estar em terapia. Use com parcimônia.
Rodável com: python sage_axiom.py
Requisitos: TensorFlow >= 2.8
"""

import tensorflow as tf
import numpy as np
import logging

# ========================= Paladin Mode Modules (SAGE14-FX v5.0) =========================

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
        return self.d2(h)

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

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
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
        self.blocks = tf.keras.Sequential([
            FractalEncoder(dim),
            FractalBlock(dim),
            FractalBlock(dim),
            FractalBlock(dim),
            tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')
        ])

    def call(self, x):
        return self.blocks(x)

# ========================= Chorus Modules (SAGE-6) =========================

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
        self.cycle = tf.Variable(tf.zeros([dim]), trainable=False)
        self.projector = tf.keras.layers.Dense(dim, activation='tanh')

    def call(self, x):
        emotion = self.projector(x)
        self.cycle.assign(0.8 * self.cycle + 0.2 * emotion)
        return self.cycle

class SymbolicContradictionHarvester(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query = tf.keras.layers.Dense(dim)
        self.memory = tf.Variable(tf.zeros([dim]), trainable=False)

    def call(self, x):
        q = self.query(x)
        self.memory.assign(0.95 * self.memory + 0.05 * q)
        return q - self.memory

class ReflexiveObserver(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.meta = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x):
        return x * self.meta(x)

class HesitationCore(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.uncertainty_proj = tf.keras.layers.Dense(dim, activation='tanh')
        self.conflict_gate = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, contradiction):
        doubt = self.uncertainty_proj(contradiction)
        gate = self.conflict_gate(doubt)
        hesitant_output = x * (1 - gate) + doubt * gate
        return hesitant_output

# ========================= Fused Model: SageAxiom =========================
class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        # === Paladin Components ===
        self.early_proj = tf.keras.layers.Conv2D(hidden_dim, 1, activation='relu')
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(2)
        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.longterm = LongTermMemory(memory_size=128, embedding_dim=hidden_dim)
        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.pain_system = TaskPainSystem(hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)
        self.projector = tf.keras.layers.Conv2D(hidden_dim, 1)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(10, 1)
        ])
        self.refiner = OutputRefinement(hidden_dim)
        self.gate_scale = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
        self.doubt = DoubtModule(hidden_dim)
        self.fallback = tf.keras.layers.Conv2D(10, 1)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # === Chorus Components ===
        self.chorus_encoder = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.synthesizer = SpectralSynthesizer(hidden_dim)
        self.crystallizer = IdentityCrystallizer(hidden_dim)
        self.crystal = AffectiveTimeCrystal(hidden_dim)
        self.harvester = SymbolicContradictionHarvester(hidden_dim)
        self.observer = ReflexiveObserver(hidden_dim)
        self.hesitator = HesitationCore(hidden_dim)
        self.chorus_decoder = tf.keras.layers.Dense(10)

        # Fusing both
        self.fusion_layer = tf.keras.layers.Conv2D(10, 1, padding='same')

    def call(self, x_seq, y_seq=None, training=False):
        batch = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        state = tf.zeros([batch, self.hidden_dim])
        self.memory.reset()

        for t in range(T):
            early = self.early_proj(x_seq[:, t])
            x = self.norm(self.encoder(early))
            x_flat = tf.keras.layers.GlobalAveragePooling2D()(x)
            x_flat = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x_flat)
            x_flat = tf.keras.layers.Dense(self.hidden_dim)(x_flat)
            out, [state] = self.agent(x_flat, [state])
            self.memory.write(out)

        memory_tensor = tf.transpose(self.memory.read_all(), [1, 0, 2])
        memory_context = self.attend_memory(memory_tensor, state)
        long_term_context = self.longterm.match_context(state)
        long_term_context = tf.reshape(long_term_context, [batch, self.hidden_dim])
        full_context = tf.concat([state, memory_context, long_term_context], axis=-1)
        context = tf.tile(tf.reshape(full_context, [batch, 1, 1, -1]), [1, 20, 20, 1])

        projected_input = self.projector(self.pos_enc(context))
        attended = self.attn(projected_input)
        chosen_transform = self.chooser(attended, hard=self.use_hard_choice)

        last_input_encoded = self.encoder(self.early_proj(x_seq[:, -1]))
        context_features = tf.concat([state, memory_context], axis=-1)
        channel_gate = self.gate_scale(context_features)
        channel_gate = tf.reshape(channel_gate, [batch, 1, 1, self.hidden_dim])
        channel_gate = tf.clip_by_value(channel_gate, 0.0, 1.0)

        gate_softmax = tf.nn.softmax(tf.concat([channel_gate, 1 - channel_gate], axis=-1), axis=-1)
        chosen_weight = gate_softmax[..., :self.hidden_dim]
        last_weight = gate_softmax[..., self.hidden_dim:]
        blended = chosen_weight * chosen_transform + last_weight * last_input_encoded

        for _ in range(2):
            refined = self.attn(blended)
            blended = tf.nn.relu(blended + refined)

        paladin_logits = self.decoder(blended)
        refined_logits = self.refiner(paladin_logits)
        doubt_score = self.doubt(blended)
        conservative_logits = self.fallback(blended)
        paladin_output = 0.5 * refined_logits + 0.5 * conservative_logits

        # Chorus path (process last frame average as vector)
        last_frame = x_seq[:, -1]  # [B, H, W, C]
        chorus_input = tf.reduce_mean(last_frame, axis=[1, 2])
        x_encoded = self.chorus_encoder(chorus_input)
        spectral = self.synthesizer(x_encoded)
        identity = self.crystallizer(spectral)
        affect = self.crystal(identity)
        contradiction = self.harvester(affect)
        reflective = self.observer(contradiction)
        hesitant = self.hesitator(reflective, contradiction)
        chorus_output = self.chorus_decoder(hesitant)

        chorus_broadcast = tf.reshape(chorus_output, [batch, 1, 1, 10])
        chorus_broadcast = tf.tile(chorus_broadcast, [1, 20, 20, 1])

        fused = tf.concat([paladin_output, chorus_broadcast], axis=-1)
        final_logits = self.fusion_layer(fused)

        if y_seq is not None:
            expected_broadcast = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            expected_broadcast = tf.reshape(expected_broadcast, tf.shape(final_logits))
            pain, gate, exploration, alpha = self.pain_system(final_logits, expected_broadcast)
            base_loss = tf.reduce_mean(tf.square(expected_broadcast - final_logits))
            sym_loss = compute_auxiliary_loss(tf.nn.softmax(final_logits))
            trait_loss = self.pain_system.compute_trait_loss(final_logits, expected_broadcast)
            total_loss = base_loss + sym_loss + trait_loss
            self.loss_tracker.update_state(total_loss)

        return final_logits

    @property
    def metrics(self):
        return [self.loss_tracker]
