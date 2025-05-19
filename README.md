# 🧠 ARC-2025 Solver com SageAxiom

Este repositório contém uma tentativa honesta (e um tanto desesperada) de resolver desafios do dataset **Abstraction and Reasoning Corpus (ARC)** usando uma arquitetura neural personalizada chamada `SageAxiom`. Este modelo combina elementos de visão computacional, embeddings de linguagem via BERT e mecanismos de votação entre múltiplas instâncias de modelos para gerar saídas em tarefas de transformação visual.

## 📦 Estrutura do Projeto

```
├── core.py                # Definição do modelo SageAxiom (onde a mágica acontece)
├── metrics_utils.py       # Funções para análise de performance e geração de gráficos
├── runtime_utils.py       # Funções utilitárias de logging, padding e temporização
├── sage_dabate_loop.py    # Loop de debate entre modelos para gerar consenso
├── neural_blocks.py       # Blocos de rede reutilizáveis (encoders, atenção, refinadores)
├── arc-agi_test_challenges.json  # Dataset de entrada (não incluso no repo por motivos legais)
└── main.py (ou equivalente)       # Script principal de treinamento e avaliação
```

## 🧪 Como funciona?

1. **Pré-processamento:** Cada par input/output das tarefas ARC é convertido em tensores fixos (30x30 com one-hot para 10 classes).
2. **Treinamento:** São treinadas múltiplas instâncias (padrão: 5) do modelo `SageAxiom`, cada uma com checkpoints, early stopping e redução de LR adaptativa.
3. **Arquitetura:** O `SageAxiom` é um monstro gentil com:

   * BERT congelado para processar prompts em linguagem natural
   * Attention over memory e módulos de escolha baseada em hipóteses
   * Refinamento por convoluções + fallback conservador
   * Um sabor suave de autoatenção e GRUs, como toda rede moderna gostaria de ser
4. **Inferência por Votação:** Cada task é resolvida por um **debate** entre modelos. Se 2 ou mais concordarem com uma resposta, ela é considerada "aceita". Caso contrário, seguimos em frente como se nada tivesse acontecido.
5. **Análise:** Gráficos de treino, matriz de confusão, e estatísticas por task são geradas no final. Ah, e tem logs. Muitos logs.

## ⚙️ Requisitos

* Python 3.8+
* TensorFlow 2.x
* scikit-learn
* matplotlib, seaborn
* transformers (para BERT)
* GPU, paciência, e um desejo forte de entender por que inteligência é tão difícil de simular

Use um ambiente virtual, a menos que você goste de viver perigosamente.

```bash
pip install -r requirements.txt  # você vai precisar montar este arquivo, claro
```

## 🏁 Execução

Para treinar e avaliar o modelo:

```bash
python main.py
```

Sim, não há um `main.py` explícito, mas você é inteligente e conseguirá encontrar o ponto de entrada. Provavelmente é o primeiro script enorme lá em cima.

## 📊 Resultados

* Os modelos produzem gráficos de treinamento (`training_plot_*.png`), matrizes de confusão e relatórios por classe.
* A pontuação final é estimada em percentual de tasks resolvidas.
* Projeções otimistas de desempenho aparecem no final, junto com as tasks mais difíceis e mais longas.

## 📁 Saídas

* `checkpoints/`: Modelos salvos por rodada
* `sage_model_{i}`: Versão final de cada modelo treinado
* `history_prompts/`: Histórico de debate e decisões por task
* `submission.json`: Resultado final do modelo (boa sorte usando isso em qualquer lugar)

## 💡 Observações

* Este projeto é um tributo ao caos e à esperança de que redes neurais consigam *pensar*.
* Ele não é perfeito, nem simples, mas é honesto em sua ambição.
* Pode não resolver o ARC... mas pelo menos tenta melhor que você.

## 📜 Licença

MIT ou algo assim. Ninguém está monetizando essa tristeza.
