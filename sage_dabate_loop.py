import json
import tensorflow as tf
from runtime_utils import log, pad_to_shape
from collections import defaultdict

def conversational_loop(models, input_grid, max_rounds=3):
    """
    Recebe modelos SageAxiom treinados e realiza um debate iterativo.
    Cada modelo propõe uma saída baseada no grid de entrada e texto.
    O vencedor é determinado por votação majoritária.
    """
    def generate_response(model, prompt):
        x = tf.convert_to_tensor([pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
        x_onehot = tf.one_hot(x, depth=10, dtype=tf.float32)
        if isinstance(prompt, str):
            prompt = [prompt]
        y_pred = model.from_prompt_and_grid(prompt, x_onehot)
        return tf.argmax(y_pred["logits"][0], axis=-1).numpy().tolist()

    prompt_text = f"Input grid:\n{json.dumps(input_grid)}"
    log("[INFO] Iniciando debate com múltiplas rodadas")
    log(prompt_text)

    all_responses = []
    for round_num in range(1, max_rounds + 1):
        log(f"[INFO] Rodada {round_num} iniciada")
        responses = []
        for i, model in enumerate(models):
            try:
                output = generate_response(model, prompt_text)
                log(f"[INFO] Modelo {i+1} produziu uma saída com sucesso na rodada {round_num}")
                log(f"[DEBUG] Output do modelo {i+1}: {output}")
                responses.append(output)
            except Exception as e:
                log(f"[ERRO] Modelo {i+1} falhou ao gerar resposta: {e}")
                responses.append(None)

        valid_responses = [r for r in responses if r is not None]

        round_entry = {
            "candidates": responses,
            "votes": [],
            "winner": None
        }

        def count_votes(candidates):
            votes = defaultdict(int)
            for c in candidates:
                key = json.dumps(c)
                votes[key] += 1
            most_common = max(votes.items(), key=lambda x: x[1])
            return json.loads(most_common[0]), most_common[1] >= 2

        if valid_responses:
            voted_output, success = count_votes(valid_responses)
            round_entry["votes"] = [json.dumps(r) for r in valid_responses]

            if success:
                winner_idx = responses.index(voted_output)
                round_entry["winner"] = winner_idx
                all_responses.append(round_entry)

                log(f"[INFO] Votação encerrada com maioria na rodada {round_num}")
                log(f"[RESULTADO] Output vencedor: {voted_output}")
                return {
                    "output": voted_output,
                    "success": True,
                    "rounds": round_num,
                    "history": all_responses
                }

        all_responses.append(round_entry)
        log("[WARN] Nenhuma resposta válida recebida nesta rodada")

    log("[INFO] Debate finalizado sem maioria")
    return {
        "output": None,
        "success": False,
        "rounds": max_rounds,
        "history": all_responses
    }
