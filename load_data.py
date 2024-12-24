#%%

from pathlib import Path
import tensorflow as tf

img_height = 200
img_width = 200
num_classes = 7
data_path = "data1"
batch_size = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
 data_path,
  validation_split=0.2, 
  labels = "inferred",
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=2)
# %%
list_ds = tf.data.Dataset.list_files(str(Path(data_path)/'*/*'), shuffle=False)
# %%
