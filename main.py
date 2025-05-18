import os
import json
import time
import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from core import SageAxiom
from layers import TaskEncoder
import llm_driver
from functions import *
from tensorflow.python.trackable.base import Trackable
import shutil

# === Hiperparâmetros e limites ===
EPOCHS = 1
TARGET_TASKS = 21
EXPECTED_HOURS = 1 / 3  # Tempo esperado para completar as tasks
TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60
SECONDS_PER_TASK = (TIME_LIMIT_MINUTES * 60) / TARGET_TASKS

# === Carregamento de dados ===
with open("arc-agi_test_challenges.json") as f:
    tasks = json.load(f)

log("[INFO] Preparando dados do SageAxiom...")
X_train_all, y_train_all = [], []
for task in tasks.values():
    for pair in task["train"]:
        input_grid = pad_to_shape(
            tf.convert_to_tensor(pair["input"], dtype=tf.int32))
        output_grid = pad_to_shape(
            tf.convert_to_tensor(pair["output"], dtype=tf.int32))
        X_train_all.append(input_grid)
        y_train_all.append(output_grid)

X_all = tf.stack(X_train_all)
y_all = tf.stack(y_train_all)
X_all_onehot = tf.one_hot(X_all, depth=10)
X_train, X_val, y_train, y_val = train_test_split(
    X_all_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42
)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

# === Treinamento ===
model = SageAxiom(hidden_dim=128, use_hard_choice=False)
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001), loss=None, metrics=[])
model(X_train[:1])
train_start = time.time()
history = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=EPOCHS, verbose=1)
training_duration = profile_time(train_start, "Tempo de treinamento")

plot_history(history)
y_val_pred = tf.argmax(model(X_val, training=False)["logits"], axis=-1).numpy()
plot_confusion(y_val.numpy(), y_val_pred)

# === Avaliação ===
submission_dict = defaultdict(list)
correct_tasks = 0
total_tasks = 0
task_times = {}
attempts_per_task = {}

for task_id, task in list(tasks.items())[:TARGET_TASKS]:
    task_start = time.time()
    input_grid = task["train"][0]["input"]
    expected_output = task["train"][0]["output"]

    input_tensor = tf.one_hot(tf.convert_to_tensor(pad_to_shape(
        tf.convert_to_tensor(input_grid, dtype=tf.int32))), depth=10, dtype=tf.float32)
    output_tensor = tf.one_hot(tf.convert_to_tensor(pad_to_shape(
        tf.convert_to_tensor(expected_output, dtype=tf.int32))), depth=10, dtype=tf.float32)

    print("\n--- Atributos rastreados ---")

    for name, val in model.__dict__.items():
        try:
            print(f"{name}: {type(val)}")
        except Exception as e:
            print(f"{name}: ERROR {e}")
        # Builda com dummy input
    export_task_embedding(task_id, hidden_dim=128)

    feedback = None
    attempt = 0
    success = False
    predicted = None
    historyPrompt = []
    log(f"[INFO] Avaliando task {task_id} ({total_tasks + 1}/{TARGET_TASKS})")

    while (time.time() - task_start) < SECONDS_PER_TASK and not success:
        codes = llm_driver.prompt_beam_llm(
            input_grid,
            llm_driver.prompt_template,
            beam_width=5,
            feedback=feedback
        )

        for code in codes:
            result = run_code(code, input_grid)
            if result["success"] and compare_outputs(result["output"], expected_output):
                log("[INFO] Qwen acertou 🎯 (beam search)")
                predicted = result["output"]
                success = True
                break
            else:
                x = tf.convert_to_tensor(
                    [pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
                x_onehot = tf.one_hot(x, depth=10, dtype=tf.float32)
                y_pred = model(x_onehot, z_task=z_task, training=False)
                fallback_output = tf.argmax(
                    y_pred["logits"][0], axis=-1).numpy().tolist()

                if compare_outputs(fallback_output, expected_output):
                    log("[INFO] SageAxiom acertou (fallback + z_task)")
                    predicted = fallback_output
                    success = True
                    break
                else:
                    feedback = describe_diff(input_grid, fallback_output)
                    historyPrompt.append(
                        {"attempt": attempt + 1, "feedback": feedback, "code": code})
            attempt += 1

    attempts_per_task[task_id] = attempt + 1
    if success:
        correct_tasks += 1
        for t in task["test"]:
            test_input = t["input"]
            try:
                test_code = llm_driver.prompt_llm(
                    test_input, llm_driver.prompt_template)
                test_result = run_code(test_code, test_input)
                if test_result["success"]:
                    submission_dict[task_id].append(test_result["output"])
                    continue
            except:
                pass
            x_test = tf.convert_to_tensor(
                [pad_to_shape(tf.convert_to_tensor(test_input, dtype=tf.int32))])
            x_onehot_test = tf.one_hot(x_test, depth=10, dtype=tf.float32)
            y_pred_test = model(x_onehot_test, z_task=z_task, training=False)
            pred_test = tf.argmax(
                y_pred_test["logits"][0], axis=-1).numpy().tolist()
            submission_dict[task_id].append(pred_test)
    os.makedirs("history_prompts", exist_ok=True)
    with open(f"history_prompts/{task_id}.json", "w") as f:
        json.dump(historyPrompt, f, indent=2)

    task_times[task_id] = profile_time(task_start, f"Task {task_id}")
    total_tasks += 1

# === Resultados finais ===
with open("submission.json", "w") as f:
    json.dump(submission_dict, f, ensure_ascii=False)

with open("per_task_times.json", "w") as f:
    json.dump(task_times, f, indent=2)

plot_attempts_stats(task_times, attempts_per_task)

log(f"[INFO] Matches corretos: {correct_tasks}/{total_tasks}")
score = (correct_tasks / total_tasks) * 100
log(f"[INFO] Score estimado: {score:.2f}%")
projection = (correct_tasks / 250) * 100
log(f"[INFO] Projeção final aproximada: {projection:.2f}%")
log("[INFO] Pipeline encerrado.")
