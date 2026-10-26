from pathlib import Path
import tensorflow as tf
import kagglehub

RANDOM_KEY=26
SIZE_IMAGE=224
BATCH_SIZE = 45
EPOCHS = 45
AUTOTUNE = tf.data.AUTOTUNE
EARLY_STOPPING_PATIENCE = 23

DATA_DIR = Path(kagglehub.dataset_download("karapy/neurovision") + '/Data')