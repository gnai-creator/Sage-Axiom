# core.py

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from layers import *
from functools import lru_cache

# Carrega e congela o modelo BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
for layer in bert_model.layers:
    layer.trainable = False

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        self.token_embedding = TokenEmbedding(vocab_size=10, dim=hidden_dim)
        self.early_proj = tf.keras.layers.Conv2D(hidden_dim, 1, activation='relu')
        self.text_proj = tf.keras.layers.Dense(self.hidden_dim)
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.rotation = LearnedRotation(hidden_dim)

        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)

        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)

        self.projector = tf.keras.layers.Conv2D(hidden_dim, 1)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(10, 1)
        ])
        self.refiner = OutputRefinement(hidden_dim)
        self.fallback = tf.keras.layers.Conv2D(10, 1)
        self.gate_scale = tf.keras.layers.Dense(hidden_dim, activation='tanh', name="gate_scale")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        self.refine_weight = self.add_weight(
            name="refine_weight", shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=False
        )

        self.pool_dense1 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(0.3)
        ])

    @lru_cache(maxsize=512)
    def cached_embed_text(self, prompt):
        inputs = bert_tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)
        outputs = bert_model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embed = self.text_proj(cls_token)
        tf.print("[BERT] text_embed.norm():", tf.norm(embed))
        return embed

    def embed_text(self, prompt):
        return self.cached_embed_text(tuple(prompt))

    def from_prompt_and_grid(self, text_prompt, x_seq):
        text_embed = self.embed_text(text_prompt)
        return self(x_seq, text_embed=text_embed, training=False)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, y_seq=y, training=True)
            loss = outputs["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        outputs = self(x, y_seq=y, training=False)
        loss = outputs["loss"]
        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    def call(self, x_seq, y_seq=None, training=False, text_embed=None):
        if x_seq.shape.rank != 4:
            raise ValueError("Esperado input de shape [batch, height, width, 10]")

        batch = x_seq.shape[0] or tf.shape(x_seq)[0]
        state = tf.zeros([batch, self.hidden_dim])

        xt = self.token_embedding(x_seq)
        xt = self.pos_enc(xt)
        xt = self.rotation(xt)
        xt = self.early_proj(xt)
        xt = self.encoder(xt, training=training)
        xt = self.norm(xt, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(xt)
        x_flat = self.pool_dense1(x_flat)

        if text_embed is not None:
            x_flat += text_embed

        out, [state] = self.agent(x_flat, [state])
        memory_tensor = tf.expand_dims(out, axis=0)
        memory_context = self.attend_memory(memory_tensor, state)
        full_context = tf.concat([state, memory_context], axis=-1)
        context = tf.reshape(full_context, [batch, 1, 1, -1])
        context = tf.tile(context, [1, 30, 30, 1])

        projected = self.projector(context)
        attended = self.attn(projected)
        chosen = self.chooser(attended, hard=self.use_hard_choice)

        last_xt = self.token_embedding(x_seq)
        last_xt = self.pos_enc(last_xt)
        last_xt = self.rotation(last_xt)
        last_xt = self.early_proj(last_xt)
        last_xt = self.encoder(last_xt, training=training)

        channel_gate = self.gate_scale(full_context)
        channel_gate = tf.reshape(channel_gate, [batch, 1, 1, self.hidden_dim])
        channel_gate = tf.tile(channel_gate, [1, 30, 30, 1])

        blended = channel_gate * chosen + (1 - channel_gate) * last_xt
        for _ in range(2):
            refined = self.attn(blended)
            blended = tf.nn.relu(blended + refined)

        logits = self.decoder(blended, training=training)
        refined_logits = self.refiner(logits, training=training)
        conservative_logits = self.fallback(blended)

        w = tf.clip_by_value(self.refine_weight, 0.0, 1.0)
        final_logits = w * refined_logits + (1.0 - w) * conservative_logits
        final_grid = tf.argmax(final_logits, axis=-1)

        if y_seq is None:
            return {"logits": final_logits, "grid": final_grid}

        expected = tf.one_hot(tf.cast(y_seq, tf.int32), depth=10, dtype=tf.float32)
        cross_entropy = tf.keras.losses.categorical_crossentropy(expected, final_logits, from_logits=True)
        loss = tf.reduce_mean(cross_entropy)
        self.add_loss(loss)
        return {"logits": final_logits, "grid": final_grid, "loss": loss}

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]