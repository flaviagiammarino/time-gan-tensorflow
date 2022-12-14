import tensorflow as tf

def encoder_embedder(timesteps, features, hidden_dim, num_layers):
    '''
    Encoder embedder, takes as input the actual sequences and returns the actual embeddings.
    '''
    x = tf.keras.layers.Input(shape=(timesteps, features))
    for _ in range(num_layers):
        e = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(x if _ == 0 else e)
    return tf.keras.models.Model(x, e, name='encoder_embedder')


def encoder(timesteps, hidden_dim, num_layers):
    '''
    Encoder, takes as input the actual embeddings and returns the actual latent vector.
    '''
    e = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        h = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(e if _ == 0 else h)
    h = tf.keras.layers.Dense(units=hidden_dim)(h)
    return tf.keras.models.Model(e, h, name='encoder')


def decoder(timesteps, features, hidden_dim, num_layers):
    '''
    Decoder, takes as input the actual or synthetic latent vector and returns the reconstructed or synthetic sequences.
    '''
    h = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=hidden_dim, activation='relu'))(h if _ == 0 else y)
    y = tf.keras.layers.Dense(units=features)(y)
    return tf.keras.models.Model(h, y, name='decoder')


def generator_embedder(timesteps, features, hidden_dim, num_layers):
    '''
    Generator embedder, takes as input the synthetic sequences and returns the synthetic embeddings.
    '''
    z = tf.keras.layers.Input(shape=(timesteps, features))
    for _ in range(num_layers):
        e = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(z if _ == 0 else e)
    return tf.keras.models.Model(z, e, name='generator_embedder')


def generator(timesteps, hidden_dim, num_layers):
    '''
    Generator, takes as input the synthetic embeddings and returns the synthetic latent vector.
    '''
    e = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        h = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(e if _ == 0 else h)
    h = tf.keras.layers.Dense(units=hidden_dim)(h)
    return tf.keras.models.Model(e, h, name='generator')


def discriminator(timesteps, hidden_dim, num_layers):
    '''
    Discriminator, takes as input the actual or synthetic embedding or latent vector and returns the log-odds.
    '''
    h = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        p = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=hidden_dim, return_sequences=True if _ < num_layers - 1 else False))(h if _ == 0 else p)
    p = tf.keras.layers.Dense(units=1)(p)
    return tf.keras.models.Model(h, p, name='discriminator')


def simulator(samples, timesteps, features):
    '''
    Simulator, generates synthetic sequences from a Wiener process.
    '''
    z = tf.random.normal(mean=0, stddev=1, shape=(samples * timesteps, features), dtype=tf.float32)
    z = tf.cumsum(z, axis=0) / tf.sqrt(tf.cast(samples * timesteps, dtype=tf.float32))
    z = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    z = tf.reshape(z, (samples, timesteps, features))
    return z
