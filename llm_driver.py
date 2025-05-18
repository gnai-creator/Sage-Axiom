# Qwen + SageAxiom ARC Solver Notebook

import re
from llama_cpp import Llama
import json
import os
import time
from collections import defaultdict
import tensorflow as tf
import hashlib

# === Inicialização do Qwen ===

model_path = os.path.expanduser("qwen2.5-1.5b-instruct-q4_k_m.gguf")
if not os.path.exists(model_path):
    model_path = os.path.expanduser("./qwen2.5-1.5b-instruct-q4_k_m.gguf")
    print(f"Modelo GGUF não encontrado em {model_path}. Verifique o caminho.")
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
    """Gera hash curto para o conteúdo do input grid."""
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]


def extract_transform_function(code: str) -> str | None:
    """Isola e retorna somente a função transform(grid) se for segura."""
    match = re.search(
        r"(def transform\(grid\):(?:\n|.)*?)(?=^def |\Z)", code, re.MULTILINE)
    if not match:
        print("[ERRO] Não foi possível encontrar a função transform.")
        return None

    func_code = match.group(1)

    blacklist = ["input(", "open(", "os.", "subprocess",
                 "eval(", "exec(", "import ", "print("]
    for item in blacklist:
        if item in code and item not in func_code:
            print(
                f"[ERRO] Código contém instrução insegura fora da função: {item}")
            return None

    if "grid" not in func_code:
        print("[ERRO] Código não utiliza o argumento `grid`.")
        return None

    return func_code


def looks_hardcoded(code: str, task_input: list) -> bool:
    """Verifica se a saída está hardcoded no corpo da função."""
    grid_flat = [str(n) for row in task_input for n in row]
    sample = grid_flat[:6]
    snippet = "".join(sample)
    return snippet in code.replace("\n", "").replace(" ", "")


def test_transform_code(code: str) -> bool:
    """Executa a função extraída com um grid dummy e verifica se funciona."""
    try:
        scope = {}
        exec(code, scope)
        if "transform" not in scope:
            print("[ERRO] 'transform' não foi definido.")
            return False

        dummy_grid = [[1, 2], [3, 4]]
        result = scope["transform"](dummy_grid)

        if not isinstance(result, list):
            print("[ERRO] Resultado da função não é uma lista.")
            return False

        return True
    except Exception as e:
        print(f"[ERRO] Teste da função falhou: {e}")
        return False


def save_bad_code(code: str, reason: str, grid: list):
    """Salva código defeituoso com contexto e razão."""
    with open(".bad_llm.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"# Motivo: {reason}\n")
        f.write(f"# Hash input: {hash_task_input(grid)}\n")
        f.write(code.strip() + "\n")


def prompt_llm(task_input: list, prompt_template: str) -> str:
    prompt = prompt_template.format(grid=json.dumps(task_input))

    if len(prompt.split()) > 500:
        print("[WARN] Prompt muito longo, pode estourar o contexto da LLM.")

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
            print("[ERRO] Código suspeito de estar hardcoded com base no input.")
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


print("Qwen + SageAxiom driver inicializado com sanidade mínima.")
