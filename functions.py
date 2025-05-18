import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
from sklearn.metrics import confusion_matrix, classification_report
from layers import TaskEncoder
import shutil

log_filename = f"log_arc_{time.strftime('%Y%m%d_%H%M%S')}.txt"
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


def describe_diff(input_grid, output_grid):
    try:
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)
        changes = np.where(input_np != output_np)
        descriptions = []
        for i, j in zip(*changes):
            before = input_np[i][j]
            after = output_np[i][j]
            descriptions.append(f"({i}, {j}): {before} → {after}")
        if not descriptions:
            return "Nenhuma mudança identificada."
        return "Mudanças detectadas: " + "; ".join(descriptions)
    except Exception as e:
        return f"[ERRO ao gerar descrição]: {e}"


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


def plot_attempts_stats(task_times, attempts_per_task):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    tasks = list(task_times.keys())
    times = [task_times[t] for t in tasks]
    attempts = [attempts_per_task[t] for t in tasks]

    color = 'tab:blue'
    ax1.set_xlabel('Task ID')
    ax1.set_ylabel('Tempo (s)', color=color)
    ax1.bar(tasks, times, color=color, alpha=0.6, label="Tempo")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tentativas', color=color)
    ax2.plot(tasks, attempts, color=color, marker='o', label="Tentativas")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Tempo e Tentativas por Task")
    plt.xticks(rotation=45)
    plt.savefig("task_performance_overview.png")
    log("[INFO] Gráfico de performance salvo: task_performance_overview.png")


def profile_time(start, label):
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    log(f"[PERF] {label}: {int(mins)}m {int(secs)}s ({elapsed:.2f} segundos)")
    return elapsed


class InlineTaskEncoder(tf.keras.Model):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(hidden_dim, activation='tanh')
        ])
        self.input_proj = tf.keras.layers.Conv2D(
            hidden_dim, 3, padding='same', activation='relu')
        self.output_proj = tf.keras.layers.Conv2D(
            hidden_dim, 3, padding='same', activation='relu')

    def call(self, inputs):
        x_in = inputs["x_in"]
        x_out = inputs["x_out"]
        if x_in.shape != x_out.shape:
            raise ValueError("Shape mismatch entre input e output")
        x_in_proj = self.input_proj(x_in)
        x_out_proj = self.output_proj(x_out)
        x = tf.concat([x_in_proj, x_out_proj], axis=-1)
        return self.encoder(x)


def export_task_embedding(task_id, hidden_dim=128):
    encoder_model = InlineTaskEncoder(hidden_dim=hidden_dim)

    dummy_input = {
        "x_in": tf.zeros([1, 30, 30, 10], dtype=tf.float32),
        "x_out": tf.zeros([1, 30, 30, 10], dtype=tf.float32),
    }
    encoder_model(dummy_input)

    @tf.function(input_signature=[{
        "x_in": tf.TensorSpec([None, 30, 30, 10], tf.float32),
        "x_out": tf.TensorSpec([None, 30, 30, 10], tf.float32),
    }])
    def serving_fn(inputs):
        return encoder_model(inputs)

    export_path = f"task_embeddings/{task_id}"
    shutil.rmtree(export_path, ignore_errors=True)
    os.makedirs(export_path, exist_ok=True)

    tf.saved_model.save(
        encoder_model,
        export_dir=export_path,
        signatures={"serving_default": serving_fn}
    )
