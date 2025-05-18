import os
import time
import json
import datetime
import traceback
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from core import SageAxiom
import llm_driver

# === Logging ===
log_filename = f"log_arc_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)


def log(msg):
    print(msg)
    logging.info(msg)


def pad_to_shape(tensor, target_shape=(30, 30)):
    pad_height = target_shape[0] - tf.shape(tensor)[0]
    pad_width = target_shape[1] - tf.shape(tensor)[1]
    return tf.pad(tensor, paddings=[[0, pad_height], [0, pad_width]], constant_values=0)


def run_code(code: str, input_matrix: list) -> dict:
    scope = {}
    try:
        exec(code, scope)
        if "transform" not in scope:
            raise ValueError("Código não define 'transform'")
        result = scope["transform"](input_matrix)
        return {"success": True, "output": result}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(limit=1)}


def compare_outputs(predicted, expected) -> bool:
    try:
        return np.array_equal(np.array(predicted), np.array(expected))
    except Exception:
        return False


def plot_history(history):
    plt.figure(figsize=(10, 5))
    for key in history.history:
        plt.plot(history.history[key], label=key)
    plt.title("SageAxiom Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    log("[INFO] Plot do treinamento salvo: training_plot.png")


def plot_confusion(y_true, y_pred):
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(10)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title("SageAxiom Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    log("[INFO] Matriz de confusão salva: confusion_matrix.png")

    report = classification_report(
        y_true_flat, y_pred_flat, labels=list(range(10)), output_dict=True)
    with open("per_class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo: per_class_metrics.json")


def profile_time(start, label):
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    log(f"[PERF] {label}: {int(mins)}m {int(secs)}s ({elapsed:.2f} segundos)")
    return elapsed


# === Execução ===
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    EPOCHS = 40
    TARGET_TASKS = 21
    EXPECTED_HOURS = 1
    TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60
    SECONDS_PER_TASK = (TIME_LIMIT_MINUTES * 60) / TARGET_TASKS

    with open("arc-agi_test_challenges.json") as f:
        tasks = json.load(f)

    # === Treinamento ===
    log("[INFO] Carregando dados do ARC para treinamento do SageAxiom...")
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

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_all_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42
    )

    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss=None, metrics=[])
    model(X_train_final[:1])

    training_start = time.time()
    history = model.fit(X_train_final, y_train_final, validation_data=(
        X_val, y_val), epochs=EPOCHS, verbose=1)
    training_time = profile_time(training_start, "Tempo de treinamento")

    plot_history(history)
    y_val_pred = tf.argmax(model(X_val, training=False)
                           ["logits"], axis=-1).numpy()
    plot_confusion(y_val.numpy(), y_val_pred)

    # === Avaliação por task ===
    submission_dict = defaultdict(list)
    correct_tasks = 0
    total_tasks = 0
    task_times = {}

    evaluation_start = time.time()
    for task_id, task in list(tasks.items())[:TARGET_TASKS]:
        task_start = time.time()
        input_grid = task["train"][0]["input"]
        expected_output = task["train"][0]["output"]
        log(f"[INFO] Task {task_id} ({total_tasks+1}/{TARGET_TASKS})")

        predicted = None
        success = False
        feedback = None
        attempt = 0
        while (time.time() - task_start) < SECONDS_PER_TASK and not success:
            code = llm_driver.prompt_llm(
                input_grid, llm_driver.prompt_template, feedback=feedback)
            result = run_code(code, input_grid)

            if result["success"] and compare_outputs(result["output"], expected_output):
                log("[INFO] Qwen acertou 🎯.")
                predicted = result["output"]
                success = True
                break
            else:
                x = tf.convert_to_tensor(
                    [pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
                x_onehot = tf.one_hot(x, depth=10, dtype=tf.float32)
                y_pred = model(x_onehot, training=False)
                fallback_output = tf.argmax(
                    y_pred["logits"][0], axis=-1).numpy().tolist()
                if compare_outputs(fallback_output, expected_output):
                    log("[INFO] SageAxiom acertou (fallback).")
                    predicted = fallback_output
                    success = True
                    break
                else:
                    feedback = f"SageAxiom sugeriu: {fallback_output}"
            attempt += 1

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
                y_pred_test = model(x_onehot_test, training=False)
                pred_test = tf.argmax(
                    y_pred_test["logits"][0], axis=-1).numpy().tolist()
                submission_dict[task_id].append(pred_test)
        else:
            log("[WARN] Nenhuma solução correta para esta task.")

        task_elapsed = profile_time(task_start, f"Tarefa {task_id}")
        task_times[task_id] = task_elapsed
        total_tasks += 1

    evaluation_time = profile_time(
        evaluation_start, "Tempo de avaliação total")

    # === Resultados ===
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission_dict, f, ensure_ascii=False)

    with open("per_task_times.json", "w", encoding="utf-8") as f:
        json.dump(task_times, f, indent=2)

    log(f"[INFO] Matches corretos: {correct_tasks}/{total_tasks}")
    score = (correct_tasks / total_tasks) * 100
    projection = (correct_tasks / 250) * 100
    log(f"[INFO] Score estimado: {score:.2f}%")
    log(
        f"[INFO] Projeção final aproximada (base 250 tasks): {projection:.2f}%")
    log("[INFO] Pipeline finalizado.")
    log(f"[INFO] Log completo salvo em: {log_filename}")