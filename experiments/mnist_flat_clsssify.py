from argparse import ArgumentParser
from tqdm import tqdm
import ai
import tensorflow as tf

def main(args):
    # todo: load mnist dataset
    train_ds, val_ds = ai.datasets.mnist()

    # todo: create and optimize model (add regularization like dropout and batch normalization)
    model = ai.models.image.FlatImageClassifier(10)

    # todo: create optimizer (optional: try with learning rate decay)
    optimizer = tf.optimizers.Adam(0.001)

    # todo: define query function
    def query(images, classes, training):
        model1_output = model(images, training=training)
        model1_loss = ai.losses.classification_loss(classes, model1_output)
        model1_loss = tf.reduce_mean(model1_loss)
        return model1_loss

    # todo: define train function
    def train(images, classes):
        with tf.GradientTape() as tape:
            loss = query(images, classes, True)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # todo: run training and evaluation for number or epochs (from argument parser)
    #  and print results (accumulated) from each epoch (train and val separately)
    loss = tf.metrics.Mean('classifier')
    for i in range(1000):

        loss.reset_states()
        with tqdm(total=60000) as pbar:
            for images, classes in train_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                loss_temp = train(images, classes)
                loss.update_state(loss_temp)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Train epoch')
        print(f'classifier loss: {loss.result().numpy()}')
        print('============================\n')

        loss.reset_states()
        with tqdm(total=10000) as pbar:
            for images, classes in val_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                loss_temp = query(images, classes, False)
                loss.update_state(loss_temp)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Validation epoch')
        print(f'classifier loss: {loss.result().numpy()}')
        print('============================\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
