# utils.py

import tensorflow as tf


def bounding_shape_penalty(pred_mask, true_mask):
    pred_size = tf.reduce_sum(tf.cast(pred_mask, tf.float32), axis=[1, 2])
    true_size = tf.reduce_sum(tf.cast(true_mask, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_size - true_size))


def continuity_loss(logits):
    probs = tf.nn.softmax(logits, axis=-1)

    dx = tf.square(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    dy = tf.square(probs[:, 1:, :, :] - probs[:, :-1, :, :])

    loss = tf.reduce_mean(dx) + tf.reduce_mean(dy)
    return 0.01 * loss  # Ajuste o peso


def temporal_symmetry_loss(logits_seq):
    flipped = tf.reverse(logits_seq, axis=[1])
    return tf.reduce_mean(tf.square(logits_seq - flipped))


def spatial_decay_mask(shape, decay_rate=0.1):
    h, w = shape[1], shape[2]
    yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    dist = tf.sqrt(tf.cast(yy**2 + xx**2, tf.float32))
    decay = tf.exp(-decay_rate * dist)
    decay = tf.expand_dims(decay, axis=0)
    decay = tf.expand_dims(decay, axis=-1)
    return decay


def repetition_penalty(logits):
    probs = tf.nn.softmax(logits, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    penalty = tf.reduce_mean(entropy)
    return 0.01 * penalty


def reverse_penalty(logits, expected):
    probs = tf.nn.softmax(logits, axis=-1)
    expected_mask = tf.reduce_max(expected, axis=-1, keepdims=True)
    non_target = 1.0 - expected_mask
    penalty = tf.reduce_mean(probs * non_target)
    return 0.01 * penalty


def edge_alignment_penalty(output_probs):
    # Extrai bordas
    top = output_probs[:, 0:1, :, :]
    bottom = output_probs[:, -1:, :, :]
    left = output_probs[:, :, 0:1, :]
    right = output_probs[:, :, -1:, :]

    # Soma total de probabilidade nas bordas
    edge_sum = tf.reduce_sum(top) + tf.reduce_sum(bottom) + \
        tf.reduce_sum(left) + tf.reduce_sum(right)

    # Número total de elementos considerados
    total_pixels = tf.cast(tf.size(top) + tf.size(bottom) +
                           tf.size(left) + tf.size(right), tf.float32)

    penalty = edge_sum / total_pixels
    return 0.01 * penalty


def compute_auxiliary_loss(probs):
    """
    Penaliza saídas pouco informativas — incentiva distribuições mais 'concentradas'.
    Exemplo: faz o modelo preferir [0,0,1,0,...] ao invés de distribuições planas.
    """
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    mean_entropy = tf.reduce_mean(entropy)
    return 0.01 * mean_entropy


def compute_all_losses(pred, expected, blended, pain_output):
    loss_dict = {}

    # Softmax para obter probabilidades
    probs = tf.nn.softmax(pred)

    # One-hot do esperado
    expected_broadcast = tf.one_hot(
        tf.cast(expected, tf.int32), depth=10, dtype=tf.float32
    )

    # === Losses principais ===
    loss_dict["base"] = tf.reduce_mean(tf.square(expected_broadcast - pred))
    loss_dict["symmetry"] = compute_auxiliary_loss(probs)
    loss_dict["spatial_penalty"] = tf.reduce_mean(
        tf.nn.relu(tf.reduce_sum(probs, axis=-1) - 1.0)
    ) * 0.01
    loss_dict["bbox"] = BoundingBoxDiscipline(
        penalty_weight=0.15)(probs, expected_broadcast)
    loss_dict["decay"] = tf.reduce_mean(
        probs * spatial_decay_mask(tf.shape(pred))
    ) * 0.005
    loss_dict["repeat"] = repetition_penalty(pred) * 0.001
    loss_dict["reverse"] = reverse_penalty(pred, expected_broadcast) * 0.001
    loss_dict["edge"] = edge_alignment_penalty(probs) * 0.001
    loss_dict["continuity"] = continuity_loss(pred) * 0.001

    # Máscaras binárias para comparação de formas
    pred_mask = tf.cast(
        tf.stop_gradient(tf.reduce_max(probs, axis=-1) > 0.5), tf.float32
    )
    true_mask = tf.cast(
        tf.reduce_max(expected_broadcast, axis=-1) > 0.5, tf.float32
    )
    loss_dict["shape"] = bounding_shape_penalty(pred_mask, true_mask) * 0.0001

    # Pain penalty (sim, isso é real)
    if pain_output is not None:
        adjusted_pain = tf.clip_by_value(
            tf.reshape(pain_output, [tf.shape(pred)[0], 1, 1, 1]), 0.0, 10.0
        )
        loss_dict["pain"] = tf.reduce_mean(adjusted_pain) * 0.05

    # Debug print (mas só se você for masoquista)
    for name, value in loss_dict.items():
        tf.print("\n", name, value)

    return loss_dict



class BoundingBoxDiscipline(tf.keras.layers.Layer):
    def __init__(self, penalty_weight=0.05):
        super().__init__()
        self.penalty_weight = 0.05

    def call(self, prediction_probs, expected_onehot):
        # Use classe dominante como máscara
        pred_classes = tf.argmax(prediction_probs, axis=-1)
        true_classes = tf.argmax(expected_onehot, axis=-1)

        # Considerar tudo que não é "classe de fundo"
        pred_mask = pred_classes > 0
        true_mask = true_classes > 0

        def compute_penalty(pair):
            pred, true = pair

            def safe_bbox(mask):
                coords = tf.where(mask)
                count = tf.shape(coords)[0]

                def fallback():
                    return tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)

                def compute():
                    y_min = tf.cast(tf.reduce_min(coords[:, 0]), tf.float32)
                    x_min = tf.cast(tf.reduce_min(coords[:, 1]), tf.float32)
                    y_max = tf.cast(tf.reduce_max(coords[:, 0]), tf.float32)
                    x_max = tf.cast(tf.reduce_max(coords[:, 1]), tf.float32)
                    return tf.stack([y_min, x_min, y_max, x_max])

                return tf.cond(count > 0, compute, fallback)

            p_box = safe_bbox(pred)
            t_box = safe_bbox(true)

            pred_area = (p_box[2] - p_box[0] + 1.0) * \
                (p_box[3] - p_box[1] + 1.0)
            true_area = (t_box[2] - t_box[0] + 1.0) * \
                (t_box[3] - t_box[1] + 1.0)

            area_penalty = tf.nn.relu(
                pred_area - true_area) / (true_area + 1.0)

            center_offset = tf.sqrt(
                tf.square((p_box[0] + p_box[2]) / 2.0 - (t_box[0] + t_box[2]) / 2.0) +
                tf.square((p_box[1] + p_box[3]) / 2.0 -
                          (t_box[1] + t_box[3]) / 2.0)
            ) / 20.0

            inter_ymin = tf.maximum(p_box[0], t_box[0])
            inter_xmin = tf.maximum(p_box[1], t_box[1])
            inter_ymax = tf.minimum(p_box[2], t_box[2])
            inter_xmax = tf.minimum(p_box[3], t_box[3])
            inter_area = tf.maximum(
                0.0, inter_ymax - inter_ymin + 1.0) * tf.maximum(0.0, inter_xmax - inter_xmin + 1.0)
            union_area = pred_area + true_area - inter_area + 1e-6
            iou = inter_area / union_area
            iou_penalty = 1.0 - iou

            total_penalty = area_penalty + center_offset + iou_penalty
            return tf.where(
                tf.reduce_any(true) & tf.reduce_any(pred),
                tf.tanh(total_penalty),
                0.0
            )

        penalties = tf.map_fn(
            compute_penalty, (pred_mask, true_mask),
            fn_output_signature=tf.float32
        )

        return self.penalty_weight * tf.reduce_mean(penalties)
