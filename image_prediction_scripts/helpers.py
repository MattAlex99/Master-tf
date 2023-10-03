import tensorflow as tf
def segmentation_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss for segmentation maps.

    Args:
        y_true: True segmentation maps (ground truth), with shape (batch_size, height, width, num_classes).
        y_pred: Predicted segmentation maps, with the same shape as y_true.

    Returns:
        Cross-entropy loss.
    """
    # Flatten the true and predicted segmentation maps
    y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

    # Compute the cross-entropy loss
    loss = tf.keras.losses.categorical_crossentropy(y_true_flat, y_pred_flat, from_logits=True)

    return loss