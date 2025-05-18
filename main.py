import os
import time
import json
import datetime
import traceback
import numpy as np
import tensorflow as tf

from collections import defaultdict
from sklearn.model_selection import train_test_split
from core import SageAxiom
import llm_driver

import logging

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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 40
TARGET_TASKS = 21
EXPECTED_HOURS = 1
TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60
SECONDS_PER_TASK = (TIME_LIMIT_MINUTES * 60) / TARGET_TASKS

start_time = time.time()
total_tasks = 0
correct_tasks = 0
submission_dict = defaultdict(list)

log(f"[INFO] Começando execução por até {TARGET_TASKS} tasks ou {TIME_LIMIT_MINUTES} minutos (~{SECONDS_PER_TASK:.1f}s por task) às {datetime.datetime.now()}.")


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


def pad_to_shape(tensor, target_shape=(30, 30)):
    pad_height = target_shape[0] - tf.shape(tensor)[0]
    pad_width = target_shape[1] - tf.shape(tensor)[1]
    return tf.pad(tensor, paddings=[[0, pad_height], [0, pad_width]], constant_values=0)


if __name__ == "__main__":
    with open("arc-agi_test_challenges.json") as f:
        tasks = json.load(f)

    log("[INFO] Preparando dados para treinamento do SageAxiom...")
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

    X_train_final = tf.convert_to_tensor(X_train_final, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_train_final = tf.convert_to_tensor(y_train_final, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

    log("[INFO] Compilando modelo SageAxiom...")
    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss=None, metrics=[])
    model(X_train_final[:1])

    log("[INFO] Iniciando treinamento...")
    history = model.fit(X_train_final, y_train_final, validation_data=(
        X_val, y_val), epochs=EPOCHS, verbose=1)

    log("[INFO] Treinamento concluído.")
    for k, v in history.history.items():
        log(f"  {k}: {[round(float(f), 4) for f in v]}")

    log("[INFO] Começando avaliação de tarefas...")
    for i, task_id in enumerate(tasks.keys()):
        if total_tasks >= TARGET_TASKS or (time.time() - start_time) > TIME_LIMIT_MINUTES * 60:
            log("[INFO] Tempo esgotado ou tarefas completas.")
            break

        task = tasks[task_id]
        input_grid = task["train"][0]["input"]
        expected_output = task["train"][0]["output"]

        log(f"[INFO] Task ID: {task_id} ({total_tasks + 1}/{TARGET_TASKS})")

        task_start_time = time.time()
        task_timeout = SECONDS_PER_TASK
        predicted = None
        success = False
        feedback = None
        attempt = 0

        while (time.time() - task_start_time) < task_timeout and not success:
            log(f"[INFO] Tentativa #{attempt + 1} com Qwen...")
            code = llm_driver.prompt_llm(
                input_grid, llm_driver.prompt_template, feedback=feedback)
            result = run_code(code, input_grid)

            if result["success"] and compare_outputs(result["output"], expected_output):
                log("[INFO] Qwen acertou 🎯.")
                predicted = result["output"]
                success = True
                break
            else:
                log("[INFO] Qwen falhou. Chamando SageAxiom para feedback...")
                x = tf.convert_to_tensor(
                    [pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))], dtype=tf.int32)
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
                    feedback = log(f"""
                    A tentativa falhou. SageAxiom sugeriu a seguinte transformação:
                    {fallback_output}
                    log("[INFO] Feedback gerado para nova tentativa com Qwen.""")

            attempt += 1

        if success:
            correct_tasks += 1
            log("[INFO] Processando testes para task correta...")
            for t in task["test"]:
                test_input = t["input"]
                try:
                    code_test = llm_driver.prompt_llm(
                        test_input, llm_driver.prompt_template)
                    result_test = run_code(code_test, test_input)
                    if result_test["success"]:
                        submission_dict[task_id].append(result_test["output"])
                        continue
                except:
                    pass

                x_test = tf.convert_to_tensor(
                    [pad_to_shape(tf.convert_to_tensor(test_input, dtype=tf.int32))], dtype=tf.int32)
                x_onehot_test = tf.one_hot(x_test, depth=10, dtype=tf.float32)
                y_pred_test = model(x_onehot_test, training=False)
                pred_test = tf.argmax(
                    y_pred_test["logits"][0], axis=-1).numpy().tolist()
                submission_dict[task_id].append(pred_test)
        else:
            log("[INFO] Task falhou completamente. Tristeza profunda.")

        total_tasks += 1

    log("[INFO] Salvando resultados...")
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission_dict, f, ensure_ascii=False)

    log("[INFO] Submissão salva: submission.json")
    log(f"[INFO] Tasks processadas: {total_tasks}")
    log(f"[INFO] Matches corretos: {correct_tasks}")
    if total_tasks > 0:
        estimated_score = correct_tasks / total_tasks * 100
        log(
            f"[INFO] Estimativa de score (em {total_tasks} tarefas): {estimated_score:.2f}%")

        final_score = (correct_tasks / 250) * 100
        log(
            f"[INFO] Projeção final aproximada com base nas 250 tasks do ARC: {final_score:.2f}%")

    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    log(f"[INFO] Tempo total de execução: {int(mins)}m {int(secs)}s ({total_time:.2f} segundos)")
    log("[INFO] Processo encerrado.")
    print(f"Logs salvos em: {log_filename}")
