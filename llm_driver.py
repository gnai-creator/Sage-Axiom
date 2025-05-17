# llm_driver.py
# Qwen-3 GGUF (Q4_K_M) - ARC Puzzle Solver Notebook

!pip install llama-cpp-python --quiet

from llama_cpp import Llama
import json
import os

# Caminho para o modelo GGUF quantizado (Q4_K_M recomendado)
model_path = "/kaggle/input/qwen-3/gguf/32b-gguf/Qwen3-32B-Q4_K_M.gguf"

# Verifica existência do modelo
assert os.path.exists(model_path), "Modelo GGUF não encontrado. Verifique o caminho."

# Inicialização do modelo com suporte a GPU (ajuste os parâmetros se necessário)
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35,  # ajuste para caber na GPU T4
    verbose=True
)

# Prompt template simples para resolução de puzzle
prompt_template = """
Dado o seguinte grid de entrada (em JSON), escreva uma função Python que gere a saída esperada do puzzle do ARC.

Grid de entrada:
{grid}

Responda apenas com código Python. Sem explicações.
"""

# Função de inferência

def prompt_llm(task_input: list, prompt_template: str) -> str:
    prompt = prompt_template.format(grid=json.dumps(task_input))
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Você é um solucionador de puzzles visuais do ARC."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return result['choices'][0]['message']['content']

# Exemplo de uso com entrada dummy do ARC
example_input = [[1, 2], [3, 4]]
generated_code = prompt_llm(example_input, prompt_template)

print("\nGenerated Code:\n")
print(generated_code)
