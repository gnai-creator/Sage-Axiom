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

def run_task_set(task_set, model, optimizer, loss_fn, phase_name=""):
    correct_pixels = total_pixels = perfect_matches = total_tasks = 0

    for task_id, task in tqdm(task_set.items(), desc=f"{phase_name} Tasks"):
        try:
            x_seq, y_seq, test_x, raw_input, expected_output = prepare_few_shot_from_task(task, shot=5, augment=True)
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
        except Exception as e:
            print(f"Erro em {phase_name} task {task_id}: {e}")

    return correct_pixels, total_pixels, perfect_matches, total_tasks

def main():
    start_time = time.time()

    with open("data/arc-agi_training-challenges.json", "r") as f:
        train_tasks = json.load(f)
    with open("data/arc-agi_evaluation-challenges.json", "r") as f:
        eval_tasks = json.load(f)
    with open("data/arc-agi_test_challenges.json", "r") as f:
        test_tasks = json.load(f)

    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    phases = [
        ("Training", train_tasks),
        ("Evaluation", eval_tasks),
        ("Testing", test_tasks),
    ]

    grand_total_correct = grand_total_pixels = grand_total_matches = grand_total_tasks = 0

    for name, tasks in phases:
        correct, total, matches, num_tasks = run_task_set(tasks, model, optimizer, loss_fn, name)
        print(f"\n{name} Results:")
        if total:
            print(f"Pixel Accuracy: {correct / total:.4%}")
        if num_tasks:
            print(f"Perfect Match Accuracy: {matches / num_tasks:.4%}")

        grand_total_correct += correct
        grand_total_pixels += total
        grand_total_matches += matches
        grand_total_tasks += num_tasks

    elapsed = time.time() - start_time
    print(f"\n=== Global Metrics ===")
    if grand_total_pixels:
        print(f"Total Pixel Accuracy: {grand_total_correct / grand_total_pixels:.4%}")
    if grand_total_tasks:
        print(f"Total Perfect Match Accuracy: {grand_total_matches / grand_total_tasks:.4%}")

    print(f"\n⏱️ Total time elapsed: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}min ({elapsed:.2f} seconds)")

if __name__ == "__main__":
    main()
