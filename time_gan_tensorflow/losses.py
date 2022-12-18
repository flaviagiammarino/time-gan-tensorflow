import tensorflow as tf

@tf.function
def mean_squared_error(y_true, y_pred):
    '''
    Mean squared error, used for calculating the supervised loss and the reconstruction loss.
    '''
    loss = tf.keras.losses.mean_squared_error(y_true=tf.expand_dims(y_true, axis=-1), y_pred=tf.expand_dims(y_pred, axis=-1))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


@tf.function
def binary_crossentropy(y_true, y_pred):
    '''
    Binary cross-entropy, used for calculating the unsupervised loss.
    '''
    loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
    return tf.reduce_mean(loss)


