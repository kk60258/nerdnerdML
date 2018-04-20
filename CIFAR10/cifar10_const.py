
__all__ = ['IMAGE_SIZE',
           'NUM_CLASSES',
           'NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
           'NUM_EXAMPLES_PER_EPOCH_FOR_EVAL',
           'DATA_URL',
           'MOVING_AVERAGE_DECAY',
           'NUM_EPOCHS_PER_DECAY',
           'LEARNING_RATE_DECAY_FACTOR',
           'INITIAL_LEARNING_RATE',
           'batch_size',
           'max_steps',
           'log_device_placement',
           'log_frequency']
# Global constants describing the CIFAR-10 data set.
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

batch_size = 128
max_steps = 100
log_frequency = 10
log_device_placement = False