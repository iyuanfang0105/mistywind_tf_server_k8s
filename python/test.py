import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


NUM_CLASSES = 10
INPUT_FEATURE = 'image'


def load_data():
    # Load training and eval data
    (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()
    # train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(train_labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(eval_labels, dtype=np.int32)
    #
    # reshape images
    # To have input as an image, we reshape images beforehand.
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)

    return train_data, train_labels, eval_data, eval_labels


def input_fn(train_data, train_labels, eval_data, eval_labels):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=4,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    return train_input_fn, eval_input_fn

def build_model(feature_columns, model_dir):
    # Create the Estimator
    training_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=100)

    classifier = tf.estimator.DNNClassifier(
        config=training_config,
        feature_columns=feature_columns,
        hidden_units=[256, 32],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=NUM_CLASSES,
        dropout=0.1)

    return classifier


def train(model, input_fn):
    model.train(input_fn=input_fn)
    return model


def evaluate(model, input_fn):
    eval_results = model.evaluate(input_fn=input_fn)
    print(eval_results)


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    :return: ServingInputReciever
    """
    reciever_tensors = {
        # The size of input image is flexible.
        INPUT_FEATURE: tf.placeholder(tf.float32, [None, None, None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        INPUT_FEATURE: tf.image.resize_images(reciever_tensors[INPUT_FEATURE], [28, 28]),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)


if __name__ == '__main__':
    train_data, train_labels, eval_data, eval_labels = load_data()
    # train_input_fn, eval_input_fn = input_fn(train_data, train_labels, eval_data, eval_labels)
    # feature_columns = [tf.feature_column.numeric_column(INPUT_FEATURE, shape=[28, 28, 1])]
    # model = build_model(feature_columns, './model')
    # model = train(model, train_input_fn)
    # evaluate(model, eval_input_fn)
    #
    # # Save the model
    # model.export_savedmodel('./model/pb',
    #                              serving_input_receiver_fn=serving_input_receiver_fn)
    #


    print(train_data[0])