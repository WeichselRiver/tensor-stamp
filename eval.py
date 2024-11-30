#%%
import numpy as np
import tensorflow as tf

img_height = 128
img_width = 128

model = tf.keras.models.load_model('model.keras')

#%%
img = tf.keras.utils.load_img(
    "Mio20.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ['Mio1', 'Mio10', 'Mio20', 'Tsd500']

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# %%
