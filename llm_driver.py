import random
import hashlib
import difflib
import json
import os
import re
from llama_cpp import Llama

# === Inicialização do modelo Qwen ===
model_path = os.path.expanduser("qwen2.5-1.5b-instruct-q4_k_m.gguf")
if not os.path.exists(model_path):
    model_path = os.path.expanduser("./qwen2.5-1.5b-instruct-q4_k_m.gguf")
    print(f"Modelo GGUF não encontrado em {model_path}.")
assert os.path.exists(model_path), "Modelo GGUF não encontrado."

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=16,
    verbose=False
)

prompt_template = """
Given the following input grid (in JSON), write a Python function named `transform(grid)` that produces the expected ARC puzzle output.

Input grid:
{grid}

Respond only with Python code. No explanations.
"""


def hash_task_input(grid: list) -> str:
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]


def extract_transform_function(code: str) -> str | None:
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


def looks_hardcoded(code: str, task_input: list) -> bool:
    grid_flat = [str(n) for row in task_input for n in row]
    sample = "".join(grid_flat[:6])
    return sample in code.replace("\n", "").replace(" ", "")


def test_transform_code(code: str) -> bool:
    try:
        scope = {}
        exec(code, scope)
        if "transform" not in scope:
            print("[ERRO] Função 'transform' não está definida.")
            return False
        result = scope["transform"]([[1, 2], [3, 4]])
        return isinstance(result, list)
    except Exception as e:
        print(f"[ERRO] Teste falhou: {e}")
        return False


def save_bad_code(code: str, reason: str, grid: list):
    with open(".bad_llm.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"# Motivo: {reason}\n")
        f.write(f"# Hash input: {hash_task_input(grid)}\n")
        f.write(code.strip() + "\n")


def prompt_llm(task_input: list, prompt_template: str, feedback: str = None, history: list = None) -> str:
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
Você é um solucionador de puzzles visuais baseados em grids do ARC.

Cada grid é uma matriz 2D de inteiros representando pixels coloridos.

## Tarefa
Dado um grid de entrada, escreva uma função Python chamada `transform(grid)` que retorna o grid de saída esperado.

## Contra-exemplo (não siga este modelo):
Input:
{json.dumps(example_grid["input"])}

Output esperado:
{json.dumps(example_grid["expected_output"])}

Função incorreta (exemplo apenas para ilustrar lógica, não para copiar):
```python
def transform(grid):
    {example_grid["code"]}
```

## Sua tarefa
Agora, resolva a transformação a seguir:

Input grid:
{json.dumps(task_input)}

Não repita a lógica usada acima. Gere uma função nova e criativa baseada no input fornecido.
"""

    seen_codes = set()
    if history:
        prompt += "\n## Histórico de tentativas anteriores:\n"
        for h in history[-3:]:
            combined_hash = hashlib.sha256(
                (str(task_input) + h['code'].strip()).encode()).hexdigest()
            seen_codes.add(combined_hash)
            prompt += f"\nTentativa #{h['attempt']}\nCódigo gerado:\n{h['code']}\nFeedback recebido:\n{h['feedback']}\n"

    if feedback:
        prompt += f"\n## Feedback mais recente do SageAxiom:\n{feedback}\n"

    prompt += """
Evite repetir padrões anteriores, especialmente soluções que apenas rotacionam ou espelham a entrada como:
- `return [row[::-1] for row in reversed(grid)]`
- `return list(map(list, zip(*grid[::-1])))`

A função deve ser genérica, operar com base em padrões lógicos ou visuais, e não deve conter valores fixos vindos do input.
Não inclua `input()`, `print()` ou exemplos de uso.
Escreva apenas a função Python válida chamada `transform(grid)`.
"""

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um especialista em lógica visual e puzzles do ARC."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
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

        if not test_transform_code(function_code):
            save_bad_code(
                function_code, "Erro ao executar transform()", task_input)
            raise ValueError("transform(grid) falhou no teste.")

        return function_code

    except Exception as e:
        print(f"[ERRO] Falha ao gerar código válido do LLM: {e}")
        print(f"[ERRO] Prompt enviado (início):\n{prompt[:500]}...")
        return "def transform(grid): return grid"


def prompt_beam_llm(task_input: list, prompt_template: str, beam_width: int = 3, feedback: str = None):
    candidates = []
    seen = set()
    history = []
    for i in range(beam_width * 3):
        try:
            current_feedback = f"Tentativa anterior falhou. Tente uma abordagem diferente da tentativa #{i+1}."
            code = prompt_llm(task_input, prompt_template,
                              feedback=current_feedback, history=history)
            code_clean = code.strip()
            sim_threshold = 0.95
            is_similar = any(difflib.SequenceMatcher(
                None, code_clean, other).ratio() > sim_threshold for other in seen)
            if not is_similar:
                seen.add(code_clean)
                candidates.append(code_clean)
            if len(candidates) >= beam_width:
                break
            history.append({"attempt": i+1, "code": code_clean,
                           "feedback": current_feedback})
        except Exception:
            continue
    return candidates
