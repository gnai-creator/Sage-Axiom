# Axiom: ARC Generalization Engine 🧠🔩

> *A transformer-esque hallucination engine for the ARC-2025 challenge, now featuring token embeddings, temporal self-loathing, and even more regret per parameter.*

---

## 🧬 Overview

**Axiom** is a multi-stage encoder-decoder model designed for solving the Abstraction and Reasoning Corpus (ARC) tasks with few-shot adaptation. It now includes learned token embeddings and a temporal symmetry loss to guilt the model into consistency over time.

---

## 🧱 Architecture Highlights

* **Input Encoding:**

  * 20×20 integer grids representing color IDs (0–9).
  * Now mapped through a `TokenEmbedding` layer to a dense continuous space, because one-hot was getting lonely.

* **Token Embedding:**

  * `Embedding(10 → hidden_dim)`, allowing the model to learn semantic priors over token IDs like a big boy.

* **Temporal Symmetry Loss:**

  * Because if your output sequence isn’t symmetric over time, what even *is* the point of sequential data?
  * Penalizes inconsistency between forward and reverse output sequences.

* **Early Projection:**

  * 1×1 `Conv2D` to project token embeddings to feature maps.

* **Encoder:**

  * Composed of fractal/residual blocks, all painfully handcrafted.

* **Memory Modules:**

  * `EpisodicMemory`: Collects representations across time like a nostalgic hoarder.
  * `LongTermMemory`: Fixed-size, non-learnable slab for associative recall.

* **Attention:**

  * Multi-head self-attention, applied wherever context might be ignored.
  * Also included: attention over memory, because even memories deserve to be seen.

* **ChoiceHypothesisModule:**

  * Evaluates multiple potential transformations.
  * Learns to pick one (or softly blend them all like an indecisive artist).
  * Entropy loss punishes fence-sitting.

* **TaskPainSystem:**

  * Computes per-sample "pain" based on prediction error.
  * Learns to gate exploration, alpha blending, and self-worth accordingly.
  * Updates historical threshold via `ThresholdMemory`.

* **Blending and Refinement:**

  * Blends current input with memory-driven hallucinations.
  * Uses a learnable mix of `refiner` and `fallback` predictions.
  * Tries not to panic.

* **Loss Functions:**

  * Pixel-wise mean square error.
  * Bounding box alignment (`BoundingBoxDiscipline`).
  * Symmetry regularization.
  * Trait loss (confidence, curiosity, resilience, etc.)
  * Temporal symmetry loss across logits over time.
  * Miscellaneous internal pain signals.

---

## 🔧 Training

Yes, it trains. Here's how:

```python
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(x_seq, y_seq, training=True)
        loss = loss_fn(y_seq[:, -1], logits) + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

The longer you train, the more painful and accurate it gets. It’s like therapy, but with gradients.

---

## 📈 Metrics

* `Pixel Accuracy`: Fraction of correctly predicted pixels.
* `Perfect Match`: If every pixel matches, you win.
* Logged metrics include:

  * `Pain`
  * `Adjusted Pain`
  * `Exploration Gate`
  * `Alpha`
  * `BBox Penalty`
  * `Temporal Symmetry Loss` (because self-consistency matters)

---

## 🗂 Files

* `core.py`: Main model (`SageAxiom`) class.
* `layers.py`: All custom layers, including memories, pain, attention, etc.
* `train_axiom.py`: Example training loop.
* `LICENSE.txt`: [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/).
* `README.md`: You’re reading it. Whoa.

---

## 📜 License

**CC BY-ND 4.0** – You may use, share, and cite this repo. But don’t remix, extend, or pretend you wrote it. That’s what the model’s for.

---

## 🧪 Notes

* `TokenEmbedding` replaces brittle one-hot encoding with learned representations.
* `Temporal Symmetry Loss` keeps the model’s timeline spiritually aligned.
* Losses and gates are emotionally unstable, but statistically reliable.

> *“The Axiom doesn’t solve ARC problems. It copes with them.”*
