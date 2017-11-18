import tensorflow as tf
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

script_dir = os.path.dirname(os.path.realpath(__file__))


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 160, 120, 3])
    conv1 = tf.layers.conv2d(input_layer, 32, (5, 5), padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), 2)
    conv2 = tf.layers.conv2d(pool1, 64, (5, 5), padding="same", activation=tf.nn.relu)
    input_flat = tf.contrib.layers.flatten(conv2)
    dense1 = tf.layers.dense(input_flat, units=1024)
    dropout1 = tf.layers.dropout(dense1, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(dropout1, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    # metrics = {
    #     "accuracy": tf.metrics.accuracy(
    #         labels=labels, predictions=predictions["classes"])}
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=metrics)


def dataset_input_fn(record, is_training, batch_size, num_epochs=1):
    dataset = tf.contrib.data.TFRecordDataset(record)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/class/label':
                tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text':
                tf.FixedLenFeature([], dtype=tf.string, default_value='')
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        #image = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
        # image = tf.reshape(image, [299, 299, 1])
        image = tf.image.decode_image(
            tf.reshape(parsed['image/encoded'], shape=[]),
            3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)
        #image.set_shape([output_height, output_width, 3])

        #label = tf.cast(parsed["image/class/label"], tf.int32)
        label = tf.cast(
            tf.reshape(parsed['image/class/label'], shape=[]),
            dtype=tf.int32)

        return image, tf.one_hot(indices=label-1, depth=2)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    #Not available in 1.2 # dataset = dataset.prefetch(batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels

# sess = tf.Session()
# with sess.as_default():
#     sess.run(dataset_input_fn(script_dir + "/dataset/test/train-00000-of-00001",
#                               False,
#                               batch_size=2,
#                               num_epochs=1))
# exit(0)

for _ in range(10):
    toy_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=script_dir + "/cnn_tflayers")

    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    toy_classifier.train(
        input_fn=lambda: dataset_input_fn(script_dir + "/dataset/train-00000-of-00001",
                                          True,
                                          batch_size=10,
                                          num_epochs=10),
        #steps=1000
    )


    # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False)
    eval_results = toy_classifier.evaluate(
        input_fn=lambda: dataset_input_fn(script_dir + "/dataset/validation-00000-of-00001",
                                          False,
                                          batch_size=10))
    print(eval_results)
