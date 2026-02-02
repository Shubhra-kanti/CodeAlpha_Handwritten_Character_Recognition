import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("model/cnn_emnist_model.h5")

(ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split="test",
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features["label"].num_classes

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    label = tf.one_hot(label, num_classes)
    return image, label

ds_test = ds_test.map(preprocess).batch(64)

loss, accuracy = model.evaluate(ds_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

emnist_mapping = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4',
    5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E',
    15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O',
    25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y',
    35:'Z'
}

for images, labels in ds_test.take(1):
    predictions = model.predict(images)
    for i in range(5):
        pred_index = np.argmax(predictions[i])
        confidence = np.max(predictions[i]) * 100
        plt.imshow(images[i].numpy().reshape(28, 28), cmap="gray")
        plt.title(f"Predicted: {emnist_mapping.get(pred_index, 'Unknown')} | Confidence: {confidence:.2f}%")
        plt.axis("off")
        plt.show()
