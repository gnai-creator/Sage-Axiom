from llama_cpp import Llama
import re
import os
import json
import hashlib
import random
import numpy as np

# Corrige inicialização do modelo
model_candidates = [f for f in os.listdir('.') if f.endswith('.gguf')]
assert model_candidates, "Nenhum arquivo .gguf encontrado no diretório."
model_path = model_candidates[0]

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=12,
    n_gpu_layers=50,
    n_batch=64,
    verbose=False,
    temperature=1.0
)

def hash_task_input(grid):
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]

def extract_transform_function(code):
    match = re.search(r"(def transform\(grid\):(?:\n|.)*?)(?=^def |\Z)", code, re.MULTILINE)
    if not match:
        return None
    func_code = match.group(1)
    if "return grid" in func_code.strip():
        return None
    blacklist = ["input(", "open(", "os.", "subprocess", "eval(", "exec(", "import ", "print("]
    if any(item in code and item not in func_code for item in blacklist):
        return None
    if "grid" not in func_code:
        return None
    return func_code

def looks_hardcoded(code, task_input):
    flat_set = set(str(n) for row in task_input for n in row)
    matches = sum(1 for n in flat_set if n in code)
    return matches >= 4  # só rejeita se metade ou mais dos elementos aparecerem no código


def test_transform_code(code, test_input=None, test_output=None):
    try:
        scope = {}
        exec(code, scope)
        if "transform" not in scope:
            return False
        result = scope["transform"](test_input or [[1, 2], [3, 4]])
        if test_output is not None:
            result_arr = np.array(result)
            expected_arr = np.array(test_output)
            return result_arr.shape == expected_arr.shape and np.allclose(result_arr, expected_arr, atol=1e-2)
        return isinstance(result, list)
    except:
        return False

def save_bad_code(code, reason, grid):
    with open(".bad_llm.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"# Motivo: {reason}\n")
        f.write(f"# Hash input: {hash_task_input(grid)}\n")
        f.write(code.strip() + "\n")

def prompt_llm(task_input, feedback=None, history=None, expected_output=None):
    example_pool = [
        {"input": [[1, 2], [3, 4]], "expected_output": [[1, 0], [1, 0]], "code": "return [[cell % 2 for cell in row] for row in grid]"},
        {"input": [[5, 5], [1, 1]], "expected_output": [[0, 0], [1, 1]], "code": "return [[1 if cell == 1 else 0 for cell in row] for row in grid]"},
        {"input": [[1, 2], [3, 4]], "expected_output": [[10, 20], [30, 40]], "code": "return [[cell * 10 for cell in row] for row in grid]"},
    ]
    example = random.choice(example_pool)

    prompt = f"""
Dado um grid de entrada, escreva uma função Python `transform(grid)` que gere o grid de saída esperado.
Evite hardcode. Use transformações gerais e baseadas na estrutura dos dados.

Exemplo:
Input: {json.dumps(example['input'])}
Esperado: {json.dumps(example['expected_output'])}
```python
def transform(grid):
    {example['code']}
```

Tarefa atual:
Input: {json.dumps(task_input)}
"""
    if expected_output:
        prompt += f"\nExpected output: {json.dumps(expected_output)}"

    if history:
        prompt += f"\n# Histórico de tentativas: {len(history)}\n"
        for h in history[-3:]:
            prompt += f"\nTentativa #{h['turn']}\nCódigo:\n{h['code']}\nFeedback:\n{h['feedback']}\n"

    if feedback:
        prompt += f"\n# Feedback do modelo:\n{feedback}\n"

    prompt += """
Gere uma nova versão da função `transform(grid)` que corrija o problema anterior. NÃO repita `return grid`.
"""

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um especialista em lógica visual e puzzles do ARC."},
                {"role": "user", "content": prompt}
            ]
        )
        full_code = result['choices'][0]['message']['content']
        function_code = extract_transform_function(full_code)

        def fail_and_log(reason):
            save_bad_code(function_code or full_code, reason, task_input)
            return "def transform(grid): raise NotImplementedError('Função inválida gerada')"

        if not function_code:
            return fail_and_log("sem_funcao_valida")

        if looks_hardcoded(function_code, task_input):
            return fail_and_log("hardcoded")

        if not test_transform_code(function_code, task_input, expected_output):
            return fail_and_log("exec_fail")

        return function_code

    except Exception as e:
        save_bad_code("# ERRO DE EXECUCAO\n" + str(e), "exception", task_input)
        return "def transform(grid): raise RuntimeError('Erro de execução no LLM')"
