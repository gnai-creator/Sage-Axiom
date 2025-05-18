import tensorflow as tf
from layers import *
from utils import (
    temporal_symmetry_loss,
    compute_all_losses,
    BoundingBoxDiscipline
)


class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        if hidden_dim is None:
            raise ValueError("hidden_dim não pode ser None")
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        self.token_embedding = TokenEmbedding(
            vocab_size=10, dim=self.hidden_dim)
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
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

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
        val_loss = outputs["loss"]
        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}

    def call(self, x_seq, y_seq=None, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(
                "Esperado input de shape [batch, height, width, 10]")

        static_batch = x_seq.shape[0] or tf.shape(x_seq)[0]

        state = tf.zeros([static_batch, self.hidden_dim])
        self.memory.reset()

        xt = self.token_embedding(x_seq)
        xt = self.pos_enc(xt)
        xt = self.rotation(xt)
        xt = self.early_proj(xt)
        xt = self.encoder(xt, training=training)
        xt = self.norm(xt, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(xt)
        x_flat = self.flat_dense1(x_flat)
        x_flat = self.flat_dense2(x_flat)
        out, [state] = self.agent(x_flat, [state])
        self.memory.write(out)

        memory_tensor = tf.transpose(self.memory.read_all(), [1, 0, 2])
        memory_context = self.attend_memory(memory_tensor, state)
        long_term_context = self.longterm.match_context(state)
        long_term_context = tf.reshape(
            long_term_context, [static_batch, self.hidden_dim])

        memory_context = tf.ensure_shape(
            memory_context, [None, self.hidden_dim])
        long_term_context = tf.ensure_shape(
            long_term_context, [None, self.hidden_dim])

        full_context = tf.concat(
            [state, memory_context, long_term_context], axis=-1)
        context = tf.tile(tf.reshape(
            full_context, [static_batch, 1, 1, -1]), [1, 30, 30, 1])

        projected_input = self.projector(context)
        attended = self.attn(projected_input)
        chosen_transform = self.chooser(attended, hard=self.use_hard_choice)

        last_xt = x_seq
        last_xt = self.token_embedding(last_xt)
        last_xt = self.pos_enc(last_xt)
        last_xt = self.rotation(last_xt)
        last_input_encoded = self.encoder(
            self.early_proj(last_xt), training=training)

        context_features = tf.concat([state, memory_context], axis=-1)
        channel_gate = self.gate_scale(context_features)
        channel_gate = tf.reshape(
            channel_gate, [static_batch, 1, 1, self.hidden_dim])
        channel_gate = tf.tile(channel_gate, [1, 30, 30, 1])
        channel_gate = tf.clip_by_value(channel_gate, 0.0, 1.0)

        blended = channel_gate * chosen_transform + \
            (1.0 - channel_gate) * last_input_encoded

        for _ in range(2):
            refined = self.attn(blended)
            blended = tf.nn.relu(blended + refined)

        output_logits = self.decoder(blended)
        refined_logits = self.refiner(output_logits)
        conservative_logits = self.fallback(blended)
        w = tf.clip_by_value(self.refine_weight, 0.0, 1.0)
        final_logits = w * refined_logits + (1.0 - w) * conservative_logits

        if y_seq is None:
            return {"logits": final_logits}

        expected_broadcast = tf.one_hot(
            tf.cast(y_seq, tf.int32), depth=10, dtype=tf.float32)

        pain_output = self.pain_system(
            final_logits, expected_broadcast, blended, training=training)

        losses = compute_all_losses(
            final_logits, y_seq, blended, pain_output["pain"])
        self.last_losses = losses
        total_loss = tf.add_n(list(losses.values()))

        if training:
            logits_seq = tf.expand_dims(final_logits, axis=1)
            total_loss += 0.01 * temporal_symmetry_loss(logits_seq)

        self.add_loss(total_loss)

        return {"logits": final_logits, "loss": total_loss}

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]
