# llm_driver.py
from llama_cpp import Llama
import re
import os
import json
import difflib
import hashlib
import random

# === Inicialização do modelo Qwen ===
import glob
model_candidates = glob.glob("./*.gguf")
assert model_candidates, "Nenhum arquivo .gguf encontrado no diretório."
model_path = model_candidates[0]


llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=12,
    n_gpu_layers=50,
    n_batch=64,
    verbose=False,
)


def hash_task_input(grid):
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]


def extract_transform_function(code):
    match = re.search(
        r"(def transform\(grid\):(?:\n|.)*?)(?=^def |\Z)", code, re.MULTILINE)
    if not match:
        print("[ERRO] Função transform(grid) não encontrada.")
        return None

    func_code = match.group(1)
    blacklist = ["input(", "open(", "os.", "subprocess",
                 "eval(", "exec(", "import ", "print("]
    for item in blacklist:
        if item in code and item not in func_code:
            print(f"[ERRO] Código inseguro detectado: {item}")
            return None

    if "grid" not in func_code:
        print("[ERRO] Código não usa o argumento 'grid'.")
        return None

    return func_code


def looks_hardcoded(code, task_input):
    grid_flat = [str(n) for row in task_input for n in row]
    sample = "".join(grid_flat[:6])
    return sample in code.replace("\n", "").replace(" ", "")


def test_transform_code(code, test_input=None, test_output=None):
    try:
        scope = {}
        exec(code, scope)
        if "transform" not in scope:
            print("[ERRO] Função 'transform' não está definida.")
            return False
        result = scope["transform"](test_input or [[1, 2], [3, 4]])
        if test_output:
            return result == test_output
        return isinstance(result, list)
    except Exception as e:
        print(f"[ERRO] Teste falhou: {e}")
        return False


def save_bad_code(code, reason, grid):
    with open(".bad_llm.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"# Motivo: {reason}\n")
        f.write(f"# Hash input: {hash_task_input(grid)}\n")
        f.write(code.strip() + "\n")


def prompt_llm(task_input, feedback=None, history=None, expected_output=None):
    example_pool = [
        {"input": [[1, 2], [3, 4]], "expected_output": [[1, 0], [1, 0]],
            "code": "return [[cell % 2 for cell in row] for row in grid]"},
        {"input": [[5, 5], [1, 1]], "expected_output": [[0, 0], [
            1, 1]], "code": "return [[1 if cell == 1 else 0 for cell in row] for row in grid]"},
        {"input": [[1, 2], [3, 4]], "expected_output": [[10, 20], [30, 40]],
            "code": "return [[cell * 10 for cell in row] for row in grid]"},
    ]
    example_grid = random.choice(example_pool)

    prompt = f"""
Dado um grid de entrada, escreva uma função Python `transform(grid)` que gere o grid de saída esperado.

Exemplo incorreto:
Input: {json.dumps(example_grid["input"])}
Esperado: {json.dumps(example_grid["expected_output"])}
```python
def transform(grid):
    {example_grid["code"]}
```

Sua tarefa:
Input:
{json.dumps(task_input)}
"""

    if expected_output:
        prompt += f"\nExpected output:\n{json.dumps(expected_output)}"

    seen_codes = set()
    if history:
        prompt += "\n# Histórico:\n"
        for h in history[-3:]:
            combined_hash = hashlib.sha256(
                (str(task_input) + h['code'].strip()).encode()).hexdigest()
            seen_codes.add(combined_hash)
            attempt_num = h.get('attempt', h.get('turn', '?'))
            prompt += f"\nTentativa #{attempt_num}\nCódigo gerado:\n{h['code']}\nFeedback recebido:\n{h['feedback']}\n"

    if feedback:
        prompt += f"\n## Feedback do SageAxiom:\n{feedback}\n"

    prompt += """
Evite repetir padrões anteriores.
Escreva apenas a função Python válida chamada `transform(grid)`.
"""

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um especialista em lógica visual e puzzles do ARC."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        full_code = result['choices'][0]['message']['content']
        function_code = extract_transform_function(full_code)

        if not function_code:
            save_bad_code(full_code, "Função ausente ou insegura", task_input)
            raise ValueError("Código extraído inválido.")

        combined_hash = hashlib.sha256(
            (str(task_input) + function_code.strip()).encode()).hexdigest()
        if combined_hash in seen_codes:
            save_bad_code(
                function_code, "Código repetido detectado no beam search", task_input)
            raise ValueError("Código repetido anteriormente para esta tarefa.")

        if looks_hardcoded(function_code, task_input):
            save_bad_code(
                function_code, "Código hardcoded detectado", task_input)
            raise ValueError("Código parece hardcoded.")

        try:
            scope = {}
            exec(function_code, scope)
            if "transform" in scope:
                manual_result = scope["transform"](task_input)
                if manual_result == expected_output:
                    return function_code
        except Exception as e:
            print(f"[ERRO] Execução direta falhou: {e}")

        if not test_transform_code(function_code, task_input, expected_output):
            save_bad_code(
                function_code, "Erro ao executar transform() ou saída incorreta", task_input)
            raise ValueError(
                "transform(grid) falhou no teste ou produziu saída incorreta.")

        return function_code

    except Exception as e:
        print(f"[ERRO] Falha ao gerar código válido do LLM: {e}")
        print(f"[ERRO] Prompt enviado (início):\n{prompt[:500]}...")
        return "def transform(grid): return grid"


def prompt_beam_llm(task_input, beam_width=1, feedback=None, expected_output=None):
    candidates = []
    seen = set()
    history = []
    for i in range(beam_width):
        try:
            current_feedback = f"Falhou. Tente diferente da tentativa #{i+1}."
            code = prompt_llm(
                task_input,
                feedback=current_feedback,
                history=history,
                expected_output=expected_output
            )
            code_clean = code.strip()
            sim_threshold = 0.95
            is_similar = any(
                difflib.SequenceMatcher(
                    None, code_clean, other).ratio() > sim_threshold
                for other in seen
            )
            if not is_similar:
                seen.add(code_clean)
                candidates.append(code_clean)
                history.append({
                    "attempt": i + 1,
                    "code": code_clean,
                    "feedback": current_feedback
                })
            if len(candidates) >= beam_width:
                break
        except Exception as e:
            print(f"[ERRO] Tentativa {i+1} falhou: {e}")
            continue
    return candidates
