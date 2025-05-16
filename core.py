# core.py
import tensorflow as tf
import logging
from layers import *

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        # === Paladin Mode ===
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
        self.gate_scale = tf.keras.layers.Dense(hidden_dim, activation='sigmoid', name="gate_scale")
        self.fallback = tf.keras.layers.Conv2D(10, 1)

        # === Chorus Core ===
        self.chorus_encoder = tf.keras.layers.Dense(hidden_dim, activation='relu', name="dense_5180")
        self.crystallizer = IdentityCrystallizer(hidden_dim)
        self.harvester = SymbolicContradictionHarvester(hidden_dim)
        self.observer = ReflexiveObserver(hidden_dim)
        self.chorus_decoder = tf.keras.layers.Dense(10, name="chorus_decoder")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.flat_dense1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name="dense_5158")
        self.flat_dense2 = tf.keras.layers.Dense(self.hidden_dim, name="dense_7")
        self.final_dense = tf.keras.layers.Dense(self.hidden_dim, name="dense_5169")
        self.final_conv = tf.keras.layers.Conv2D(10, 1, padding='same', name="final_logits_conv")


    def call(self, x_seq, y_seq=None, training=False):
        batch = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        state = tf.zeros([batch, self.hidden_dim])
        self.memory.reset()

        for t in range(T):
            early = self.early_proj(x_seq[:, t])
            x = self.norm(self.encoder(early, training=training), training=training)
            x_flat = tf.keras.layers.GlobalAveragePooling2D()(x)
            x_flat = self.flat_dense1(x_flat)
            x_flat = self.flat_dense2(x_flat)
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

        output_logits = self.decoder(blended)
        refined_logits = self.refiner(output_logits)
        conservative_logits = self.fallback(blended)
        paladin_output = 0.5 * refined_logits + 0.5 * conservative_logits

        # Chorus branch (vector path)
        last_frame = x_seq[:, -1]
        chorus_input = self.final_dense(tf.reduce_mean(last_frame, axis=[1, 2]))
        x_encoded = self.chorus_encoder(chorus_input)
        identity = self.crystallizer(x_encoded)
        contradiction = self.harvester(identity)
        reflective = self.observer(contradiction)
        chorus_output = self.chorus_decoder(reflective)

        # Merge Paladin + Chorus
        chorus_broadcast = tf.reshape(chorus_output, [batch, 1, 1, 10])
        chorus_broadcast = tf.tile(chorus_broadcast, [1, 20, 20, 1])
        fused = tf.concat([paladin_output, chorus_broadcast], axis=-1)
        final_logits = self.final_conv(fused)

        if y_seq is not None:
            expected_broadcast = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            expected_broadcast = tf.reshape(expected_broadcast, tf.shape(final_logits))
            
            pain, gate, exploration, alpha = self.pain_system(final_logits, expected_broadcast, blended=blended)
        
            base_loss = tf.reduce_mean(tf.square(expected_broadcast - final_logits))
            sym_loss = compute_auxiliary_loss(tf.nn.softmax(final_logits))
            trait_loss = self.pain_system.compute_trait_loss(final_logits, expected_broadcast)
        
            chorus_loss = 0.001 * tf.reduce_mean(tf.square(chorus_output))
            self.add_loss(chorus_loss)
        
            # ⛓️ Forçar uso explícito dos pesos dessas camadas
            chorus_vars = self.chorus_encoder.trainable_variables + self.final_dense.trainable_variables
            chorus_weight_penalty = 0.001 * tf.add_n([tf.reduce_sum(tf.square(v)) for v in chorus_vars])
            self.add_loss(chorus_weight_penalty)
            crystallizer_loss = 0.001 * tf.reduce_sum(tf.square(self.crystallizer.state))
            self.add_loss(crystallizer_loss)
        
            total_loss = base_loss + sym_loss + trait_loss + tf.add_n(self.losses)
            self.loss_tracker.update_state(total_loss)
        
        return final_logits

    @property
    def metrics(self):
        return [self.loss_tracker]
