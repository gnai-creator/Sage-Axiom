import os
import json
import time
import csv
import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from core import SageAxiom
import llm_driver
from functions import *
from agent_chat import *
import matplotlib.pyplot as plt

# === Hiperparâmetros e limites ===
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 40
TARGET_TASKS = 21
EXPECTED_HOURS = 2.5
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
    learning_rate=LEARNING_RATE), loss=None, metrics=[])
model(X_train[:1])
train_start = time.time()
history = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
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
scores = []
task_ids = []


os.makedirs("history_prompts", exist_ok=True)
with open("history_prompts/history_all.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(
        csvfile, fieldnames=["task_id", "attempt", "feedback", "code"])
    csv_writer.writeheader()

    for task_id, task in list(tasks.items())[:TARGET_TASKS]:
        task_start = time.time()
        input_grid = task["train"][0]["input"]
        expected_output = task["train"][0]["output"]

        x = tf.convert_to_tensor(
            [pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
        x_onehot = tf.one_hot(x, depth=10, dtype=tf.float32)

        feedback = None
        attempt = 0
        success = False
        predicted = None
        historyPrompt = []
        log(f"[INFO] Avaliando task {task_id} ({total_tasks + 1}/{TARGET_TASKS})")

        # Loop conversacional com tempo limite
        while (time.time() - task_start) < SECONDS_PER_TASK and not success:
            result = conversational_loop(
                model=model,
                prompt_fn=prompt_from_grid,
                input_grid=input_grid,
                expected_output=expected_output
            )

            success = result["success"]
            predicted = result["predicted"]
            feedback = result["feedback"]
            attempt = result["attempt"]

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
                text_prompt = prompt_from_grid(test_input)
                y_pred_test = model.from_prompt_and_grid(
                    text_prompt, x_onehot_test)
                pred_test = tf.argmax(
                    y_pred_test["logits"][0], axis=-1).numpy().tolist()
                submission_dict[task_id].append(pred_test)

        with open(f"history_prompts/{task_id}.json", "w") as f:
            json.dump(historyPrompt, f, indent=2)

        with open(f"history_prompts/{task_id}.md", "w", encoding="utf-8") as mdfile:
            mdfile.write(f"# Histórico da Task {task_id}\n\n")
            for item in historyPrompt:
                mdfile.write(f"## Tentativa {item['attempt']}\n")
                mdfile.write("```python\n")
                mdfile.write(item['code'] + "\n")
                mdfile.write("```\n\n")
                mdfile.write(f"**Feedback:** {item['feedback']}\n\n")

        for item in historyPrompt:
            item_flat = item.copy()
            item_flat["task_id"] = task_id
            csv_writer.writerow(item_flat)

        task_times[task_id] = profile_time(task_start, f"Task {task_id}")
        total_tasks += 1
        task_ids.append(task_id)
        scores.append(int(success))

# === Resultados finais ===
with open("submission.json", "w") as f:
    json.dump(submission_dict, f, ensure_ascii=False)

with open("per_task_times.json", "w") as f:
    json.dump(task_times, f, indent=2)

plot_attempts_stats(task_times, attempts_per_task)

# === Novo: Gráficos ===


def plot_success_by_task(task_ids, scores):
    plt.figure(figsize=(10, 4))
    plt.bar(task_ids, scores, color="green")
    plt.xlabel("Task ID")
    plt.ylabel("Sucesso (1 = acerto, 0 = erro)")
    plt.title("Acertos por Task")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("history_prompts/task_success_bar.png")
    plt.close()


def plot_engagement_bar(attempts_dict):
    plt.figure(figsize=(10, 4))
    tasks = list(attempts_dict.keys())
    attempts = [attempts_dict[k] for k in tasks]
    plt.bar(tasks, attempts, color="skyblue")
    plt.xlabel("Task ID")
    plt.ylabel("Tentativas")
    plt.title("Engajamento por Task (nº de tentativas)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("history_prompts/task_engagement_bar.png")
    plt.close()


plot_success_by_task(task_ids, scores)
plot_engagement_bar(attempts_per_task)

log(f"[INFO] Matches corretos: {correct_tasks}/{total_tasks}")
score = (correct_tasks / total_tasks) * 100
log(f"[INFO] Score estimado: {score:.2f}%")
projection = (correct_tasks / 250) * 100
log(f"[INFO] Projeção final aproximada: {projection:.2f}%")
log("[INFO] Pipeline encerrado.")
