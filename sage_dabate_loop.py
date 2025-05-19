import json
import tensorflow as tf
from functions import log, pad_to_shape profile_time, plot_history, plot_attempts_stats
def triple_conversational_loop(models, input_grid):
    """
    Recebe três modelos SageAxiom treinados e realiza um debate triplo.
    Cada modelo propõe uma saída baseada no grid de entrada e texto.
    O vencedor é determinado por votação majoritária.
    Retorna a melhor saída escolhida, se houver consenso.
    """
    import random
    from collections import defaultdict

    def generate_response(model, prompt):
        x = tf.convert_to_tensor([pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
        x_onehot = tf.one_hot(x, depth=10, dtype=tf.float32)
        y_pred = model.from_prompt_and_grid(prompt, x_onehot)
        return tf.argmax(y_pred["logits"][0], axis=-1).numpy().tolist()

    prompt_text = f"Input grid:\n{json.dumps(input_grid)}"

    # Cada modelo propõe uma solução
    responses = []
    for model in models:
        try:
            output = generate_response(model, prompt_text)
            responses.append(output)
        except Exception as e:
            responses.append(None)

    # Filtra respostas válidas
    valid_responses = [r for r in responses if r is not None]

    def count_votes(candidates):
        votes = defaultdict(int)
        for c in candidates:
            key = json.dumps(c)
            votes[key] += 1
        most_common = max(votes.items(), key=lambda x: x[1])
        return json.loads(most_common[0]), most_common[1] >= 2

    if valid_responses:
        voted_output, success = count_votes(valid_responses)
    else:
        voted_output, success = None, False

    return {
        "output": voted_output,
        "success": success,
        "rounds": 1,
        "history": responses
    }