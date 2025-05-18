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
    prompt = prompt_template.format(grid=json.dumps(task_input))
    if history:
        prompt += "\n\n# Histórico de Tentativas e Resultados:\n"
        for h in history[-3:]:
            prompt += f"\nTentativa #{h['attempt']}:\nCódigo gerado:\n{h['code']}\nResultado: {h['result']}\n"

    if feedback:
        prompt += f"\n\n# Feedback mais recente do SageAxiom:\n{feedback}\n"

    prompt += "\n# Gere uma nova função transform(grid) com base nas tentativas acima."

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system",
                    "content": "Você é um solucionador de puzzles visuais do ARC."},
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
    for _ in range(beam_width):
        try:
            code = prompt_llm(task_input, prompt_template, feedback=feedback)
            if code:
                candidates.append(code)
        except Exception:
            continue
    return candidates


print("Qwen + SageAxiom driver inicializado com modo beam search disponível.")
