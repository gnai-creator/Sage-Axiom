🧠 SageAxiom
Fractal Neural Architecture for ARC-2025 Challenges
Because pixel-perfect pain builds real generalization.

✨ Visão Geral
SageAxiom é um modelo de aprendizagem profundo projetado para lidar com tarefas da competição ARC Prize 2025. Ele tenta capturar padrões visuais complexos com um sistema de memória episódica, codificação posicional 2D, refinamento de saída e um sistema punitivo de dor e disciplina... porque aparentemente "só supervisionado" não estava funcionando.

🧬 Arquitetura
early_proj: Projeção convolucional de entrada

EnhancedEncoder: Encoder fractal com blocos residuais e normalização

EpisodicMemory: Acúmulo seqüencial de embeddings (estilo RNN mas com memória externa)

LongTermMemory: Banco de vetores fixos que podem ser consultados por similaridade

MultiHeadAttentionWrapper: Atenção própria sobre o contexto

ChoiceHypothesisModule: Geração e seleção ponderada de transformações (estilo ensemble)

TaskPainSystem: Penalidade contextual baseada em softmax, entropia, e emoções simuladas (sim, sério)

OutputRefinement: Refinamento convolucional final com fallback conservador

BoundingBoxDiscipline: Penalidade baseada em área de bounding box para coibir entusiasmo fora de controle

⚙️ Treinamento
O modelo é treinado com:

Loss principal: diferença quadrática pixel a pixel (MSE)

Auxiliares: entropia de distribuição, bounding box penalty, e regulação de gate / alpha do sistema de dor

Otimizador: Adam com learning_rate=3e-4

Épocas recomendadas: entre 100–300 (depende da GPU e da paciência humana)

python
Copiar
Editar
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
🧪 Modo de Inferência
python
Copiar
Editar
# Preparar o few-shot
x_seq, y_seq, test_x, test_input, expected_output = prepare_few_shot_from_task(
    task, shot=5, augment=True
)

# Treinar
for epoch in range(300):
    with tf.GradientTape() as tape:
        logits = model(x_seq, y_seq, training=True)
        loss = loss_fn(y_seq[:, -1], logits) + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Inferência final
pred = model(test_x, training=False)
📦 Requisitos
TensorFlow 2.11+

NumPy

Matplotlib / Seaborn (para visualização)

Muita tolerância à dor

📈 Métricas
Pixel Accuracy: acurácia pixel a pixel

Perfect Match: quantas saídas foram idênticas à ground truth

Logs de atributos internos como: pain, exploration, alpha, gate, etc.

🔥 Motivação Filosófica
“Se uma rede convolucional sofre dor o suficiente, ela aprende. Ou pelo menos para de fazer besteira com a softmax.”
— Você, provavelmente, às 3 da manhã.

🧼 Licença

CC BY-ND 4.0
