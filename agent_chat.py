from llm_driver import prompt_llm
import json
from llm_driver import *

def prompt_from_grid(input_grid, history=None):
    # Você pode usar o histórico aqui se quiser enriquecer o prompt
    return f"Input grid:\n{json.dumps(input_grid)}"


def generate_code_from_qwen(input_grid, history=None, feedback=None, expected_output=None):
    return prompt_llm(
        task_input=input_grid,
        feedback=feedback,
        history=history,
        expected_output=expected_output
    )


def qwen_propose(prompt_fn, input_grid, history):
    code = generate_code_from_qwen(
        input_grid,
        history=history,
        feedback=history[-1]["feedback"] if history else None,
    )
    return code



def sage_feedback(sage_model, input_grid, expected_output, code_suggestion):
    from functions import run_code, compare_outputs, describe_diff

    result = run_code(code_suggestion, input_grid)
    if result["success"]:
        if compare_outputs(result["output"], expected_output):
            return "Perfeito. Transformação correta.", True
        else:
            diff = describe_diff(input_grid, result["output"])
            return f"Transformação falhou. Diferença: {diff}", False
    return "Erro ao executar o código sugerido.", False


def conversational_loop(model, prompt_fn, input_grid, expected_output, max_turns=3):
    history = []
    for turn in range(max_turns):
        code = qwen_propose(prompt_fn, input_grid, history)
        feedback, success = sage_feedback(
            model, input_grid, expected_output, code)
        history.append({
            "turn": turn + 1,
            "code": code,
            "feedback": feedback
        })
        if success:
            break
    return history
