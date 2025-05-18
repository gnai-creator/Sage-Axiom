import tensorflow as tf
from layers import *
from utils import (
    bounding_shape_penalty,
    continuity_loss,
    temporal_symmetry_loss,
    spatial_decay_mask,
    repetition_penalty,
    reverse_penalty,
    edge_alignment_penalty,
    compute_auxiliary_loss,
)


class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        self.token_embedding = TokenEmbedding(vocab_size=10, dim=hidden_dim)
        self.early_proj = tf.keras.layers.Conv2D(
            hidden_dim, 1, activation='relu')
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.rotation = LearnedRotation(hidden_dim)

        self.bbox_penalty = BoundingBoxDiscipline(penalty_weight=0.15)
        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.longterm = LongTermMemory(
            memory_size=128, embedding_dim=hidden_dim)
        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.pain_system = TaskPainSystem(latent_dim=hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)

        self.projector = tf.keras.layers.Conv2D(hidden_dim, 1)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(10, 1)
        ])
        self.refiner = OutputRefinement(hidden_dim)
        self.fallback = tf.keras.layers.Conv2D(10, 1)
        self.gate_scale = tf.keras.layers.Dense(
            hidden_dim, activation='tanh', name="gate_scale")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.refine_weight = self.add_weight(
            name="refine_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=False
        )

        self.flat_dense1 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu')
        self.flat_dense2 = tf.keras.layers.Dense(self.hidden_dim)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, y, training=True)
            total_loss = outputs["loss"]

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        outputs = self(x, y, training=False)
        total_loss = outputs["loss"]
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, x_seq, y_seq=None, training=False):
        if x_seq.shape.rank != 5:
            raise ValueError(
                "Esperado input de shape [batch, T, height, width, 10]")

        batch = tf.shape(x_seq)[0]
        T = x_seq.shape[1]
        if T is None:
            raise ValueError(
                "T precisa ser conhecido estaticamente para usar range(T)")

        state = tf.zeros(tf.stack([batch, self.hidden_dim]))
        self.memory.reset()

        for t in range(T):
            xt = x_seq[:, t]
            xt = self.token_embedding(xt)
            early = self.rotation(self.pos_enc(xt))
            early = self.early_proj(early)
            x = self.encoder(early, training=training)
            x = self.norm(x, training=training)

            x_flat = tf.keras.layers.GlobalAveragePooling2D()(x)
            x_flat = self.flat_dense2(self.flat_dense1(x_flat))
            out, [state] = self.agent(x_flat, [state])
            self.memory.write(out)

        memory_tensor = tf.transpose(self.memory.read_all(), [1, 0, 2])
        memory_context = self.attend_memory(memory_tensor, state)
        long_term_context = tf.reshape(self.longterm.match_context(state), [
                                       batch, self.hidden_dim])
        full_context = tf.concat(
            [state, memory_context, long_term_context], axis=-1)

        context = tf.tile(tf.reshape(
            full_context, [batch, 1, 1, -1]), [1, 30, 30, 1])
        projected_input = self.projector(context)
        attended = self.attn(projected_input)
        chosen_transform = self.chooser(attended, hard=self.use_hard_choice)

        last_xt = x_seq[:, -1]
        last_xt = self.token_embedding(last_xt)
        last_input_encoded = self.encoder(self.early_proj(
            self.rotation(self.pos_enc(last_xt))), training=training)

        channel_gate = tf.clip_by_value(tf.tile(tf.reshape(self.gate_scale(tf.concat(
            [state, memory_context], -1)), [batch, 1, 1, -1]), [1, 30, 30, 1]), 0.0, 1.0)
        blended = channel_gate * chosen_transform + \
            (1.0 - channel_gate) * last_input_encoded

        for _ in range(2):
            blended = tf.nn.relu(blended + self.attn(blended))

        output_logits = self.decoder(blended)
        refined_logits = self.refiner(output_logits)
        final_logits = tf.clip_by_value(self.refine_weight, 0.0, 1.0) * refined_logits + (
            1.0 - self.refine_weight) * self.fallback(blended)

        if y_seq is None:
            return {"logits": final_logits}

        y_last = y_seq[:, -1] if y_seq.shape.rank == 4 else y_seq
        expected_broadcast = tf.one_hot(
            tf.cast(y_last, tf.int32), depth=10, dtype=tf.float32)

        pain_output = self.pain_system(
            final_logits, expected_broadcast, blended, training=training)
        adjusted_pain = tf.clip_by_value(tf.reshape(
            pain_output["pain"], [batch, 1, 1, 1]), 0.0, 10.0)

        pixelwise_diff = tf.square(expected_broadcast - final_logits)
        base_loss = tf.reduce_mean(pixelwise_diff)
        sym_loss = compute_auxiliary_loss(tf.nn.softmax(final_logits))
        regional_penalty = 0.01 * \
            tf.reduce_mean(tf.nn.relu(
                tf.reduce_sum(final_logits, axis=-1) - 1.0))

        probs = tf.nn.softmax(final_logits / 1.5)
        bbox_loss = self.bbox_penalty(probs, expected_broadcast)
        decay_mask = spatial_decay_mask(tf.shape(final_logits))
        spread_penalty = tf.reduce_mean(probs * decay_mask) * 0.005
        repeat_penalty_val = repetition_penalty(final_logits) * 0.001
        reverse_penalty_val = reverse_penalty(
            final_logits, expected_broadcast) * 0.001
        edge_penalty_val = edge_alignment_penalty(probs) * 0.001
        cont_loss_val = continuity_loss(final_logits) * 0.001

        pred_mask = tf.cast(tf.stop_gradient(
            tf.reduce_max(probs, axis=-1) > 0.5), tf.float32)
        true_mask = tf.cast(tf.reduce_max(
            expected_broadcast, axis=-1) > 0.5, tf.float32)
        shape_loss = bounding_shape_penalty(pred_mask, true_mask) * 0.0001

        total_loss = base_loss + sym_loss + regional_penalty + bbox_loss + \
            spread_penalty + repeat_penalty_val + reverse_penalty_val + \
            edge_penalty_val + cont_loss_val + shape_loss + \
            0.05 * tf.reduce_mean(adjusted_pain)

        if training:
            logits_seq = tf.expand_dims(final_logits, axis=1)
            total_loss += 0.01 * temporal_symmetry_loss(logits_seq)

        total_loss = tf.clip_by_value(total_loss, 0.0, 100.0)
        self.add_loss(total_loss)

        return {"logits": final_logits, "loss": total_loss}

    @property
    def metrics(self):
        return [self.loss_tracker]
