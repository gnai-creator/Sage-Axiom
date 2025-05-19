# ARC-2025 Solver com SageAxiom

Este repositório apresenta uma solução baseada em redes neurais para os desafios propostos pelo dataset **Abstraction and Reasoning Corpus (ARC)**. O sistema utiliza uma arquitetura chamada `SageAxiom`, que integra componentes de visão computacional, embeddings de linguagem com BERT e um mecanismo de consenso entre múltiplas instâncias de modelos.

## Estrutura do Projeto

```
├── core.py                # Definição do modelo SageAxiom
├── metrics_utils.py       # Módulos de análise de performance
├── runtime_utils.py       # Funções auxiliares para logging e padronização
├── sage_dabate_loop.py    # Lógica de inferência baseada em votação entre modelos
├── neural_blocks.py       # Componentes modulares reutilizáveis da arquitetura neural
├── arc-agi_test_challenges.json  # Conjunto de tarefas (externo)
└── main.py                # Script principal de execução
```

## Funcionamento Geral

1. **Preparação dos Dados:** As tarefas do ARC são convertidas em tensores com tamanho fixo (30x30) usando codificação one-hot para 10 classes.
2. **Treinamento:** São treinadas múltiplas instâncias (padrão: 5) do modelo `SageAxiom` com técnicas de regularização e checkpoints automáticos.
3. **Arquitetura:** O `SageAxiom` combina:

   * Embeddings textuais com BERT (camadas congeladas)
   * Codificação posicional 2D e atenção multi-cabeça
   * GRU para memória de curto prazo
   * Mecanismo de escolha baseada em hipóteses
   * Módulo de refinamento da saída
4. **Inferência:** A saída para cada tarefa é gerada através de um ciclo de votação entre modelos, promovendo maior robustez ao sistema.
5. **Avaliação:** Métricas, matrizes de confusão e estatísticas de execução são geradas ao final da avaliação.

## Requisitos

* Python 3.8+
* TensorFlow 2.x
* scikit-learn
* matplotlib
* seaborn
* transformers (HuggingFace)
* GPU recomendada para treinamento

Instalação de dependências:

```bash
pip install -r requirements.txt
```

## Execução

Para treinar e avaliar o modelo:

```bash
python main.py
```

## Resultados

* Gráficos de histórico de treinamento: `training_plot_*.png`
* Matrizes de confusão por modelo: `confusion_matrix_*.png`
* Relatórios por classe: `per_class_metrics.json`
* Estatísticas por tarefa: `task_performance_overview.png`

## Saídas Geradas

* `checkpoints/`: Pesos intermediários salvos durante treinamento
* `sage_model_{i}/`: Modelos finais treinados
* `history_prompts/`: Registro detalhado de cada rodada de inferência
* `submission.json`: Resultado final para as tarefas

## Licença

Este projeto está licenciado sob os termos da licença **CC BY-ND 4.0**.
