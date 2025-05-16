# train_axiom.py
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns
import time

from core import SageAxiom

MAX_GRID_SIZE = 20
COLOR_DEPTH = 10

def encode_grid(grid):
    grid = np.array(grid)
    h, w = grid.shape
    padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int32)
    padded[:h, :w] = grid[:h, :w]
    return padded

def onehot_grid(grid):
    return tf.one_hot(encode_grid(grid), depth=COLOR_DEPTH, dtype=tf.float32)

def augment_grid(grid, op):
    ops = {
        "flipud": np.flipud,
        "fliplr": np.fliplr,
        "rot90": lambda x: np.rot90(x, k=1),
        "rot180": lambda x: np.rot90(x, k=2),
        "rot270": lambda x: np.rot90(x, k=3),
        "flipud_rot90": lambda x: np.rot90(np.flipud(x), k=1),
        "fliplr_rot90": lambda x: np.rot90(np.fliplr(x), k=1),
    }
    return ops[op](grid)

def prepare_few_shot_from_task(task_json, shot=3, augment=False, include_original=True):
    train_pairs = task_json['train']
    actual_shots = min(len(train_pairs), shot)
    x_seq, y_seq = [], []
    augment_ops = ["flipud", "fliplr", "rot90", "rot180", "rot270", "flipud_rot90", "fliplr_rot90"]

    for pair in train_pairs[:actual_shots]:
        input_grid = pair['input']
        output_grid = pair['output']
        if include_original:
            x_seq.append(onehot_grid(input_grid))
            y_seq.append(encode_grid(output_grid))
        if augment:
            for op in augment_ops:
                aug_input = augment_grid(input_grid, op)
                aug_output = augment_grid(output_grid, op)
                x_seq.append(onehot_grid(aug_input))
                y_seq.append(encode_grid(aug_output))

    x_seq = tf.stack(x_seq)[tf.newaxis, ...]
    y_seq = tf.stack(y_seq)[tf.newaxis, ...]

    test_pair = train_pairs[actual_shots - 1]
    test_x = onehot_grid(test_pair['input'])[tf.newaxis, tf.newaxis, ...]
    test_input = np.array(test_pair['input'])
    expected_output = encode_grid(test_pair['output'])

    return x_seq, y_seq, test_x, test_input, expected_output

def main():
    start_time = time.time()

    with open("data/arc-agi_test_challenges.json", "r") as f:
        test_tasks = json.load(f)

    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    sample_keys = list(test_tasks.keys())[:5]
    correct_pixels = total_pixels = perfect_matches = total_tasks = 0

    for task_id in tqdm(sample_keys):
        task = test_tasks[task_id]
        x_seq, y_seq, test_x, raw_input, expected_output = prepare_few_shot_from_task(task, shot=5, augment=True)

        print(f"\n== Task: {task_id} ==")
        print("x_seq shape:", x_seq.shape)
        print("y_seq shape:", y_seq.shape)

        _ = model(x_seq, y_seq, training=False)

        for epoch in range(300):
            with tf.GradientTape() as tape:
                logits = model(x_seq, y_seq, training=True)
                loss_main = loss_fn(y_seq[:, -1], logits)
                loss = loss_main + tf.add_n(model.losses)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pred = model(test_x, training=False)
        pred_img = tf.argmax(pred[0], axis=-1).numpy()

        expected_output_np = np.array(expected_output)
        correct_pixels += np.sum(pred_img == expected_output_np)
        total_pixels += expected_output_np.size
        if np.allclose(pred_img, expected_output):
            perfect_matches += 1
        total_tasks += 1

        plt.figure(figsize=(15, 4))
        plt.suptitle(f"Task {task_id}", fontsize=14)

        plt.subplot(1, 4, 1)
        plt.imshow(raw_input, cmap='tab10', vmin=0, vmax=9)
        plt.title("Input")

        plt.subplot(1, 4, 2)
        plt.imshow(expected_output_np, cmap='tab10', vmin=0, vmax=9)
        plt.title("Expected Output")

        plt.subplot(1, 4, 3)
        pred_confidence = tf.nn.softmax(pred[0], axis=-1).numpy()
        heatmap = np.max(pred_confidence, axis=-1)
        sns.heatmap(heatmap, vmin=0, vmax=1, cmap='viridis')
        plt.title("Confidence Heatmap")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_img, cmap='tab10', vmin=0, vmax=9)
        plt.title("Predicted")

        plt.show()

    elapsed = time.time() - start_time

    if total_pixels:
        print(f"Pixel Accuracy: {correct_pixels / total_pixels:.4%}")
    if total_tasks:
        print(f"Perfect Match Accuracy: {perfect_matches / total_tasks:.4%}")

    print(f"\n⏱️ Total time elapsed: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}min ({elapsed:.2f} seconds)")
  
if __name__ == "__main__":
    main()
