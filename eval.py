#%%
import numpy as np
import tensorflow as tf

img_height = 200
img_width = 200

model = tf.keras.models.load_model('model.keras')

#%%
valid_images = ["Mio20.jpg", "Mio1.jpg", "Mio10.jpg"]
for val_image in valid_images:
    img = tf.keras.utils.load_img(
    val_image, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['253_2000_blau', '254_3000_braun', '255_4000_violett', 'Mio1', 'Mio10', 'Mio20', 'Tsd500']

    print(
    f"Image {val_image} most likely belongs to class{class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence."

)
# %%

# %%
