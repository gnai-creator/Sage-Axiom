# Qwen + SageAxiom ARC Solver Notebook

# !pip install llama-cpp-python --quiet

from llama_cpp import Llama
import json
import os
import time
from collections import defaultdict
import tensorflow as tf

# === Inicialização do Qwen ===

# model_path = "/kaggle/input/qwen-3/gguf/32b-gguf/1/Qwen3-32B-Q4_K_M.gguf"
model_path = os.path.expanduser("Qwen3-14B-Q4_K_M.gguf")
if not os.path.exists(model_path):
    model_path = os.path.expanduser("./Qwen3-14B-Q4_K_M.gguf")
    print(f"Modelo GGUF não encontrado em {model_path}. Verifique o caminho.")
assert os.path.exists(model_path), "Modelo GGUF não encontrado."

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35,
    verbose=False
)

prompt_template = """
Given the following input grid (in JSON), write a Python function named `transform(grid)` that produces the expected ARC puzzle output.

Input grid:
{grid}

Respond only with Python code. No explanations.
"""


def prompt_llm(task_input: list, prompt_template: str) -> str:
    prompt = prompt_template.format(grid=json.dumps(task_input))
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um solucionador de puzzles visuais do ARC."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        code = result['choices'][0]['message']['content']
        if "def transform" not in code:
            raise ValueError("Resposta do LLM não contém 'transform'.")
        return code
    except Exception as e:
        print(f"[INFO] Erro ao gerar código LLM: {e}")
        print(f"[INFO] Resposta recebida do LLM:\n{code}")
        return "def transform(grid): return grid"

print("Ok")
