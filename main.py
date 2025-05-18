# main.py

from sklearn.model_selection import train_test_split
import time
import datetime
import json
import traceback
import numpy as np
import tensorflow as tf
from collections import defaultdict
from core import SageAxiom
import llm_driver
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_code(code: str, input_matrix: list) -> dict:
    scope = {}
    try:
        exec(code, scope)
        if "transform" not in scope:
            raise ValueError("Código não define 'transform'")
        result = scope["transform"](input_matrix)
        return {"success": True, "output": result}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(limit=1)
        }


def compare_outputs(predicted, expected) -> bool:
    try:
        return np.array_equal(np.array(predicted), np.array(expected))
    except Exception:
        return False


def pad_to_shape(tensor, target_shape=(30, 30)):
    pad_height = target_shape[0] - tf.shape(tensor)[0]
    pad_width = target_shape[1] - tf.shape(tensor)[1]
    return tf.pad(tensor, paddings=[[0, pad_height], [0, pad_width]], constant_values=0)


def time_distributed_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(
        tf.reshape(y_true, [-1, 10]),
        tf.reshape(y_pred, [-1, 10]),
        from_logits=True
    )


def debug_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=1)  # [B, 30, 30]
    y_true_onehot = tf.one_hot(
        tf.cast(y_true, tf.int32), depth=10)  # [B, 30, 30, 10]
    return tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred, from_logits=True)




if __name__ == "__main__":
    # === Parâmetros ===
    TARGET_TASKS = 20
    EXPECTED_HOURS = 0.5
    TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60
    SECONDS_PER_TASK = (TIME_LIMIT_MINUTES * 60) / TARGET_TASKS

    # === Carregar dataset ===
    with open("arc-agi_test_challenges.json") as f:
        tasks = json.load(f)

    start_time = time.time()
    total_tasks = 0
    correct_tasks = 0
    submission_dict = defaultdict(list)

    print(f"⏱️ Iniciando: {TARGET_TASKS} tasks ou {TIME_LIMIT_MINUTES:.1f} minutos (~{SECONDS_PER_TASK:.1f}s por task) as {datetime.datetime.now()}.")

    task_keys = list(tasks.keys())

    # === Preparar dados para treino do SageAxiom ===
    X_train_all = []
    y_train_all = []

    for task in tasks.values():
        for pair in task["train"]:
            input_grid = tf.convert_to_tensor(pair["input"], dtype=tf.int32)
            output_grid = tf.convert_to_tensor(pair["output"], dtype=tf.int32)

            input_padded = pad_to_shape(input_grid)
            output_padded = pad_to_shape(output_grid)

            X_train_all.append(input_padded)
            y_train_all.append(output_padded)

    X_train_all = tf.stack(X_train_all)
    y_train_all = tf.stack(y_train_all)

    X_train_onehot = tf.one_hot(X_train_all, depth=10)
    y_train_onehot = tf.one_hot(y_train_all, depth=10)

    # === Inicializar e treinar o modelo ===
    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    model.compile(optimizer='adam', loss=debug_loss)
    input_shape = (None, 1, 30, 30, 10)
    model.build(input_shape)
    X_train_onehot = tf.convert_to_tensor(X_train_onehot, dtype=tf.float32)
    X_train_onehot = tf.expand_dims(
        X_train_onehot, axis=1)  # [batch, T=1, H, W, 10]

    y_train_onehot = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)
    y_train_onehot = tf.expand_dims(
        y_train_onehot, axis=1)  # [batch, T=1, H, W, 10]
    print("X_train shape:", X_train_onehot.shape)
    print("y_train shape:", y_train_onehot.shape)
    print("Sample y_train_onehot sum:", tf.reduce_sum(y_train_onehot[0]))
    sample_pred = model(X_train_onehot[:1], training=False)
    print("Sample pred logits mean:", tf.reduce_mean(sample_pred["logits"]))
    print("Sample pred logits max:", tf.reduce_max(sample_pred["logits"]))
    if "loss" in sample_pred:
        print("Sample loss:", sample_pred["loss"])

    # y_train_int é o ground truth como índices, sem one-hot
    y_train_int = tf.convert_to_tensor(y_train_all, dtype=tf.int32)
    y_train_int = tf.expand_dims(y_train_int, axis=1)  # [batch, 1, H, W]

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_onehot, y_train_int,
        test_size=0.2,
        random_state=42
    )
    
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=40,
        verbose=1
    )



    print("Treinamento concluído. Histórico:", history.history)
    # === Loop principal ===
    for i, task_id in enumerate(task_keys):
        if total_tasks >= TARGET_TASKS or (time.time() - start_time) > TIME_LIMIT_MINUTES * 60:
            print("⏰ Tempo esgotado ou tarefas completas.")
            break

        task = tasks[task_id]
        input_grid = task["train"][0]["input"]
        expected_output = task["train"][0]["output"]

        print(f"\n🔍 Task ID: {task_id} ({total_tasks+1}/{TARGET_TASKS})")

        # Tenta com Qwen
        code = llm_driver.prompt_llm(input_grid, llm_driver.prompt_template)
        result = run_code(code, input_grid)

        success = False

        if result["success"] and compare_outputs(result["output"], expected_output):
            print("✅ Match via Qwen")
            success = True
            predicted = result["output"]
        else:
            print("❌ Falha via Qwen → tentando SageAxiom")

            x = tf.convert_to_tensor(
                [pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))], dtype=tf.int32)
            x = tf.expand_dims(x, axis=1)  # [1, 1, H, W]
            x_onehot = tf.one_hot(x, depth=10)  # [1, 1, H, W, 10]

            y_pred = model(x_onehot, training=False)
            fallback_output = tf.argmax(y_pred[0], axis=-1).numpy().tolist()

            if compare_outputs(fallback_output, expected_output):
                print("✅ Match via SageAxiom (fallback)")
                success = True
                predicted = fallback_output
            else:
                print("🛑 Falha completa")

        # Se teve sucesso em qualquer abordagem, resolver test cases
        if success:
            for t in task["test"]:
                test_input = t["input"]

                code_test = llm_driver.prompt_llm(
                    test_input, llm_driver.prompt_template)
                result_test = run_code(code_test, test_input)

                if result_test["success"]:
                    submission_dict[task_id].append(result_test["output"])
                else:
                    x_test = tf.convert_to_tensor(
                        [pad_to_shape(tf.convert_to_tensor(test_input, dtype=tf.int32))], dtype=tf.int32)
                    x_test = tf.expand_dims(x_test, axis=1)  # [1, 1, H, W]
                    x_onehot_test = tf.one_hot(
                        x_test, depth=10)  # [1, 1, H, W, 10]
                    y_pred_test = model(x_onehot_test, training=False)
                    pred_test = tf.argmax(
                        y_pred_test[0], axis=-1).numpy().tolist()
                    submission_dict[task_id].append(pred_test)

            correct_tasks += 1

        total_tasks += 1

    # === Salvar submissão ===
    with open("submission.json", "w") as f:
        json.dump(submission_dict, f)

    print(f"\n📦 Submissão salva: submission.json")
    print(f"🎯 Tasks processadas: {total_tasks}")
    print(f"✅ Matches corretos: {correct_tasks}")
