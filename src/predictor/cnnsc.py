import tensorflow as tf

EMB_SIZE = 100


def gen_conv(batch_input, out_channels, kernel_size, a):
    # [batch, in_width, in_channels] => [batch, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv1d(batch_input, out_channels, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=2, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def create_model(model_inputs, a):
    print("Generator input shape", model_inputs.shape)
    layers = []
    num_classes = 1

    # embedding: [batch, sentence_len] => [batch, sentence_len, EMB_SIZE]
    with tf.variable_scope("embedding"):
        embeddings = tf.Variable(tf.random_normal([a.vocab_size, EMB_SIZE], dtype=tf.float32))
        output = tf.nn.embedding_lookup(embeddings, model_inputs)
        layers.append(output)

    layer_specs = [
        (50, 1),  # encoder_1: [batch, sentence_len, EMB_SIZE] => [batch, 1, 50]
        (50, 2),  # encoder_2: [batch, sentence_len, EMB_SIZE] => [batch, 1, 50]
        (50, 3),  # encoder_3: [batch, sentence_len, EMB_SIZE] => [batch, 1, 50]
        (50, 4),  # encoder_4: [batch, sentence_len, EMB_SIZE] => [batch, 1, 50]
    ]

    for out_channels, kernel_size in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers))):
            convolved = gen_conv(layers[0], out_channels, kernel_size, a)
            output = batchnorm(convolved)
            rectified = lrelu(output, 0.2)
            output = tf.layers.max_pooling1d(rectified, a.max_len, 1)
            layers.append(output)

    # [batch, 1, 200]
    output = tf.concat(layers[1:], axis=2)
    if a.dropout > 0.0:
        output = tf.nn.dropout(output, keep_prob=1 - a.dropout)

    # mlp: [batch, 1, 200] => [batch, num_classes]
    with tf.variable_scope("mlp"):
        initializer = tf.random_normal_initializer(0, 0.02)
        output = tf.layers.conv1d(output, num_classes, kernel_size=1, strides=1, padding="valid", kernel_initializer=initializer)
        output = tf.squeeze(output, axis=1)
        layers.append(output)

    return layers[-1]
