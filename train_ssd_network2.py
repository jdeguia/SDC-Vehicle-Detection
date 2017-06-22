from pprint import pprint

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 4,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 8, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
              is over individual gradients. The inner list is over the gradient
              calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D and the other way.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


def main(_):
    pprint(FLAGS.__flags)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():

        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        
        with tf.device('/cpu:0'):
            global_step = slim.create_global_step()
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            optimizer = _configure_optimizer(learning_rate)


        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)
        pprint(dict(ssd_params._asdict()))

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes = \
                image_preprocessing_fn(image, glabels, gbboxes, ssd_shape)
            # Encode groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = \
                ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] + [len(ssd_anchors)] * 3

            # Training batches and queue.
            r = tf.train.batch(
                _reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            b_image, b_gclasses, b_glocalisations, b_gscores = \
                _reshape_list(r, batch_shape)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                _reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
                capacity=2 * FLAGS.num_clones)

        tower_grads=[]
        tower_losses=[]
        device_scopes = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gid in range(FLAGS.num_clones):
                b_image, b_gclasses, b_glocalisations, b_gscores = \
                    _reshape_list(batch_queue.dequeue(), batch_shape)

                with tf.device('/gpu:%d' % gid):
                    with tf.name_scope('%s_%d' % ("tower", gid)) as scope:
                        b_image, b_gclasses, b_glocalisations, b_gscores = \
                            _reshape_list(batch_queue.dequeue(), batch_shape)

                        arg_scope = ssd_net.arg_scope(weight_decay=0.0004)
                        with slim.arg_scope(arg_scope):
                            predictions, localisations, logits, end_points = \
                                ssd_net.net(b_image, is_training=True)

                        # Add loss functions.
                        ssd_net.losses(logits, localisations,
                                       b_gclasses, b_glocalisations, b_gscores,
                                       label_smoothing=0.0)

                        all_losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        sum_loss = tf.add_n(all_losses)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data
                        grads = optimizer.compute_gradients(sum_loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_losses.append(sum_loss)
                        device_scopes.append(scope)

        total_loss = tf.add_n(tower_losses, name='total_loss')
        grads = _average_gradients(tower_grads)
        apply_gradient_ops = optimizer.apply_gradients(grads, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(0.999, global_step)
        averages_op = ema.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_ops, averages_op)
        train_tensor = control_flow_ops.with_dependencies([train_op], total_loss,
                                                          name='train_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        slim.learning.train(
            train_tensor,
            logdir='logs',
            master='',
            is_chief=True,
            init_fn=None,
            summary_op=None,
            number_of_steps=1,
            log_every_n_steps=10,
            save_summaries_secs=60,
            saver=saver,
            save_interval_secs=60,
            sync_optimizer=None,
            session_config=config)

        import pdb; pdb.set_trace()

        # summary_op = tf.summary.merge_all()

        # import time
        # with tf.Session(config=config) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     tf.train.start_queue_runners(sess=sess)

        #     train_writer = tf.summary.FileWriter('./logs', sess.graph)
        #     for i in range(10000):
        #         start = time.time()
        #         gl,_, summary_str = sess.run([global_step, train_tensor, summary_op])
        #         end = time.time()
        #         print(end - start)
        #         train_writer.add_summary(summary_str, gl)
        #         # if gl % 10 == 0: print(gl)
        #     quit()


if __name__ == '__main__':
    tf.app.run()
