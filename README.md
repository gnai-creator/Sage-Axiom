# Axiom: ARC Generalization Engine 🧠🔩

> *A model architecture designed for solving the Abstraction and Reasoning Corpus (ARC) tasks, with a few-shot inner-loop adaptation approach and zero respect for your patience.*

---

## 🧬 Overview

**Axiom** is a multi-stage encoder-decoder model crafted to tackle ARC-2025 challenge problems with few-shot learning. It leans heavily on episodic memory, speculative transformation selection, and something that vaguely resembles emotional regulation, if you're into anthropomorphizing.

---

## 🧱 Architecture Highlights

* **Input Encoding:**

  * One-hot encoding of 20×20 grids with 10 color channels.
  * Optional geometric augmentation (rotation, flipping, deep existential dread).

* **Early Projection:**

  * `Conv2D(1×1)` layer to project inputs into `hidden_dim` space.

* **Encoder:**

  * Combination of fractal blocks and residual layers for spatial pattern encoding.
  * Deep, unnecessarily opinionated.

* **Memory Modules:**

  * `EpisodicMemory` stores temporal embeddings.
  * `LongTermMemory` serves as a global, frozen attention slab.

* **Attention:**

  * Multi-head attention applied in multiple phases.
  * Contextual alignment with past states and long-term bias.

* **ChoiceHypothesisModule:**

  * Projects multiple `hypotheses` over transformed features.
  * A soft or hard selector weights which transformation to apply.
  * Penalizes indecision with entropy loss. Because commitment matters.

* **TaskPainSystem + ThresholdMemory:**

  * Calculates task difficulty through "pain" metrics.
  * Uses `ThresholdMemory` to adapt its ethical judgment dynamically, based on historical trauma (average adjusted pain).
  * Balances exploration and assertiveness using noise, sigmoid gates, and more emotional instability than you'd think reasonable.

* **Blending:**

  * Learns to combine last-step encoder output with transformed memory context.
  * Gating is modulated by a `TaskPainSystem` because every decision hurts.

* **Refinement:**

  * Outputs pass through two refinement paths: confident (`refiner`) and fallback (`conservative`).
  * A learnable scalar mixes the outputs.

* **Losses:**

  * Standard pixel-wise diff loss.
  * Bounding box regularization (`BoundingBoxDiscipline`).
  * Symmetry penalty.
  * Trait-based auxiliary loss from the `TaskPainSystem`.
  * Historical threshold loss from `ThresholdMemory`, because past pain is never forgotten.

---

## 🔧 Training

The model can be trained task-by-task with a simple few-shot regime:

```python
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(x_seq, y_seq, training=True)
        loss_main = loss_fn(y_seq[:, -1], logits)
        loss = loss_main + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Performance improves with longer training, aggressive augmentation, and general willingness to wait.

---

## 📈 Metrics

* `Pixel Accuracy`: Fraction of pixels correctly predicted.
* `Perfect Match Accuracy`: Entire grid match across all pixels.

Also logs `Pain`, `Gate`, `Exploration`, `Threshold`, and other emotionally unstable diagnostics.

---

## 🗂 Files

* `core.py`: Contains the main Axiom model class.
* `layers.py`: All auxiliary modules, attention, pain system, memory, etc.
* `train_axiom.py`: Orchestrates task loading, training loop, visualization.

---

## 💡 Usage Tips

* Train with `shot=3 to 5`, `augment=True` for good generalization.
* Suggested: 100–300 epochs per task (or until GPU smokes).
* Keep logs of `Gate`, `Adjusted Pain`, `Threshold`, and `BBox Loss` to debug generalization failures.

---

## 📜 License

**CC BY-ND 4.0** – You’re free to use, share, and admire this work as long as:

* You credit the author(s).
* You don’t remix, transform, or build upon it.

Violators will be visited by the `TaskPainSystem`, and it’s not pretty.

---

## 🐍 Disclaimer

This project assumes a TensorFlow 2.x environment with no internet access, no external pretraining, and an unreasonable tolerance for pain.

> *“Axiom doesn’t solve ARC. It endures it.”*
