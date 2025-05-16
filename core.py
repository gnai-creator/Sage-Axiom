# core.py
import tensorflow as tf
import logging
from layers import *

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        # Paladin Components
        self.early_proj = tf.keras.layers.Conv2D(hidden_dim, 1, activation='relu')
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(2)
        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.longterm = LongTermMemory(memory_size=128, embedding_dim=hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)
        self.projector = tf.keras.layers.Conv2D(hidden_dim, 1)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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
        return attended

    @property
    def metrics(self):
        return [self.loss_tracker]
