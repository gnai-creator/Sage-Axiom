# Qwen + SageAxiom ARC Solver Notebook

# !pip install llama-cpp-python --quiet

from llama_cpp import Llama
import json
import os
import time
from collections import defaultdict
import tensorflow as tf

# === Inicialização do Qwen ===

model_path = "/kaggle/input/qwen-3/gguf/32b-gguf/1/Qwen3-32B-Q4_K_M.gguf"
assert os.path.exists(model_path), "Modelo GGUF não encontrado."

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35,
    verbose=False
)

prompt_template = """
Dado o seguinte grid de entrada (em JSON), escreva uma função Python chamada `transform(grid)` que gere a saída esperada do puzzle do ARC.

Grid de entrada:
{grid}

Responda apenas com código Python. Sem explicações.
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
        print(f"⚠️ Erro ao gerar código LLM: {e}")
        print(f"📤 Resposta recebida do LLM:\n{code}")
        return "def transform(grid): return grid"

print("Ok")
