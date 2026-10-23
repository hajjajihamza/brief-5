from pathlib import Path
import tensorflow as tf
import kagglehub

RANDOM_KEY=26
SIZE_IMAGE=224
BATCH_SIZE = 45
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

DATA_DIR = Path(kagglehub.dataset_download("hamzahajjaji/brief-5-dataset") + '/Data')