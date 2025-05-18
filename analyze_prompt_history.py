# analyze_prompt_history.py

import os
import json
from collections import Counter, defaultdict

PROMPT_DIR = "history_prompts"


def carregar_history():
    data = {}
    for filename in os.listdir(PROMPT_DIR):
        if filename.endswith(".json"):
            task_id = filename.replace(".json", "")
            with open(os.path.join(PROMPT_DIR, filename), encoding="utf-8") as f:
                data[task_id] = json.load(f)
    return data


def analisar_tentativas(histories):
    tentativas_por_task = {}
    erros_mais_comuns = Counter()
    falhas_qwen = 0
    salvamentos_sage = 0

    for task_id, tentativas in histories.items():
        tentativas_por_task[task_id] = len(tentativas)
        for tentativa in tentativas:
            if tentativa["result"] != "Output incorreto":
                erros_mais_comuns[tentativa["result"]] += 1
        if len(tentativas) > 0:
            if tentativas[-1]["result"] != "Output incorreto":
                falhas_qwen += 1
                salvamentos_sage += 1  # presumimos fallback salvou

    return tentativas_por_task, erros_mais_comuns, falhas_qwen, salvamentos_sage


def exibir_estatisticas(tentativas_por_task, erros_mais_comuns, falhas_qwen, salvamentos_sage):
    total_tasks = len(tentativas_por_task)
    total_tentativas = sum(tentativas_por_task.values())
    media_tentativas = total_tentativas / total_tasks if total_tasks else 0

    print(f"Total de tasks analisadas: {total_tasks}")
    print(f"Média de tentativas por task: {media_tentativas:.2f}")
    print(f"Tasks em que Qwen falhou completamente: {falhas_qwen}")
    print(f"Fallbacks bem-sucedidos do SageAxiom: {salvamentos_sage}")
    print("\nTop 5 erros mais comuns do Qwen:")
    for err, count in erros_mais_comuns.most_common(5):
        print(f"- {err}: {count} ocorrências")

    mais_tentativas = sorted(tentativas_por_task.items(),
                             key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 tasks com mais tentativas:")
    for task, count in mais_tentativas:
        print(f"- {task}: {count} tentativas")


if __name__ == "__main__":
    histories = carregar_history()
    tentativas_por_task, erros_mais_comuns, falhas_qwen, salvamentos_sage = analisar_tentativas(
        histories)
    exibir_estatisticas(tentativas_por_task, erros_mais_comuns,
                        falhas_qwen, salvamentos_sage)
