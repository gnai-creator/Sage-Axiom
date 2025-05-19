from llm_driver import prompt_llm
import json
from llm_driver import *
from functions import *
import json



def prompt_from_grid(input_grid, history=None):
    return f"Input grid:\n{json.dumps(input_grid)}"


def generate_code_from_qwen(input_grid, history=None, feedback=None, expected_output=None):
    return prompt_llm(
        task_input=input_grid,
        feedback=feedback,
        history=history,
        expected_output=expected_output
    )


def qwen_propose(prompt_fn, input_grid, history):
    feedback = history[-1]["feedback"] if history else None
    code = generate_code_from_qwen(
        input_grid,
        history=history,
        feedback=feedback
    )
    log(f"""[QWEN] Código gerado na tentativa {len(history)+1}:
{code}""")
    return code

def sage_feedback(model, input_grid, expected_output, code_suggestion):
    result = run_code(code_suggestion, input_grid)
    if result["success"]:
        if compare_outputs(result["output"], expected_output):
            log("[SAGE] Transformação correta identificada.")
            return "Perfeito. Transformação correta.", True
        else:
            diff = describe_diff(input_grid, result["output"])
            log(f"[SAGE] Transformação falhou. Diferença: {diff}")
            return f"Transformação falhou. Diferença: {diff}", False
    log("[SAGE] Erro ao executar o código sugerido.")
    return "Erro ao executar o código sugerido.", False

def conversational_loop(model, prompt_fn, input_grid, expected_output, max_turns=3):
    history = []
    success = False
    feedback = ""

    for turn in range(max_turns):
        log(f"[LOOP] Iniciando tentativa {turn+1}...")
        code = qwen_propose(prompt_fn, input_grid, history)
        feedback, success = sage_feedback(model, input_grid, expected_output, code)
        history.append({
            "turn": turn + 1,
            "code": code,
            "feedback": feedback
        })
        if success:
            log(f"[LOOP] Sucesso alcançado na tentativa {turn+1}.")
            break
        else:
            log(f"[LOOP] Tentativa {turn+1} falhou.")

    predicted = run_code(history[-1]["code"], input_grid)["output"] if success else None

    return {
        "success": success,
        "history": history,
        "code": history[-1]["code"] if success else None,
        "feedback": feedback,
        "attempt": turn + 1,
        "predicted": predicted
    }
