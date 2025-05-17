# main.py
import json
import traceback
import numpy as np

def run_code(code: str, input_matrix: list) -> dict:
    """
    Executa um código contendo uma função 'transform(grid)' e retorna o resultado.
    """
    scope = {}
    try:
        exec(code, scope)
        if "transform" not in scope:
            raise ValueError("Código não define 'transform'")
        result = scope["transform"](input_matrix)
        return {"success": True, "output": result}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(limit=1)
        }

def compare_outputs(predicted, expected) -> bool:
    try:
        return np.array_equal(np.array(predicted), np.array(expected))
    except Exception:
        return False

def test_single_task(task_json: dict, code: str) -> dict:
    """
    Testa o código gerado pelo LLM em um par (input, output) do ARC.
    """
    input_grid = task_json["train"][0]["input"]
    expected_output = task_json["train"][0]["output"]
    
    result = run_code(code, input_grid)
    if result["success"]:
        match = compare_outputs(result["output"], expected_output)
        return {
            "match": match,
            "output": result["output"]
        }
    else:
        return {
            "match": False,
            "error": result["error"]
        }

if __name__ == "__main__":
    # Demonstração: usar JSON de exemplo
    with open("example_task.json", "r") as f:
        task = json.load(f)

    # Exemplo de código gerado por LLM
    code = """
def transform(grid):
    return [[cell for cell in row] for row in grid]
"""

    result = test_single_task(task, code)
    print("Resultado:", json.dumps(result, indent=2))
