import re
import json
import os
import hashlib
from llama_cpp import Llama

# === Inicialização do modelo Qwen ===

model_path = os.path.expanduser("qwen2.5-1.5b-instruct-q4_k_m.gguf")
if not os.path.exists(model_path):
    model_path = os.path.expanduser("./qwen2.5-1.5b-instruct-q4_k_m.gguf")
    print(f"Modelo GGUF não encontrado em {model_path}.")
assert os.path.exists(model_path), "Modelo GGUF não encontrado."

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
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

# === Utilitários ===


def hash_task_input(grid: list) -> str:
    """Gera hash curto do input grid para rastrear problemas."""
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]


def extract_transform_function(code: str) -> str | None:
    """Extrai a função transform(grid) de maneira segura."""
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
    """Detecta se o código está hardcoded com base no input."""
    grid_flat = [str(n) for row in task_input for n in row]
    sample = "".join(grid_flat[:6])
    code_clean = code.replace("\n", "").replace(" ", "")
    return sample in code_clean


def test_transform_code(code: str) -> bool:
    """Executa a função com um grid fictício para validar."""
    try:
        scope = {}
        exec(code, scope)
        if "transform" not in scope:
            print("[ERRO] Função 'transform' não está definida.")
            return False

        dummy_result = scope["transform"]([[1, 2], [3, 4]])
        return isinstance(dummy_result, list)
    except Exception as e:
        print(f"[ERRO] Teste falhou: {e}")
        return False


def save_bad_code(code: str, reason: str, grid: list):
    """Salva código rejeitado junto com a razão e hash."""
    with open(".bad_llm.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"# Motivo: {reason}\n")
        f.write(f"# Hash input: {hash_task_input(grid)}\n")
        f.write(code.strip() + "\n")


def prompt_llm(task_input: list, prompt_template: str, feedback: str = None) -> str:
    prompt = prompt_template.format(grid=json.dumps(task_input))
    if feedback:
        prompt += f"\n\nThe previous attempt failed. Here is some feedback:\n{feedback}\nTry again."

    if len(prompt.split()) > 500:
        print("[WARN] Prompt pode estar excedendo o limite de contexto.")

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system",
                    "content": "You are a Python expert solving ARC grid puzzles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        full_code = result['choices'][0]['message']['content']
        function_code = extract_transform_function(full_code)

        if not function_code:
            save_bad_code(full_code, "Função ausente ou insegura", task_input)
            raise ValueError("Código extraído inválido.")

        if looks_hardcoded(function_code, task_input):
            save_bad_code(
                function_code, "Hardcoded output detectado", task_input)
            raise ValueError("Código parece hardcoded.")

        if not test_transform_code(function_code):
            save_bad_code(
                function_code, "Erro ao executar transform()", task_input)
            raise ValueError("transform(grid) falhou no teste.")

        return function_code

    except Exception as e:
        print(f"[ERRO] Falha no prompt para LLM: {e}")
        return "def transform(grid): return grid"


print("Qwen + SageAxiom driver inicializado com sanidade mínima.")
