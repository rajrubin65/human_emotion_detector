from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def esc(code):
    return f'\033[{code}m'


class EmotionDetectEnginee():
    def __init__(self,image_path='',img=None) :
        self.model = load_model('models/imageclassifier.h5')
        self.image_path = image_path
        self.img_height = 180
        self.img_width = 180
        self.img = img
        self.class_names = ['angry', 'happy', 'sad']


    def prediction(self):
        if self.img == None:
            img = tf.keras.utils.load_img(
                self.image_path, target_size=(self.img_height, self.img_width)
            )
        else:
            img = img
        # img.show()
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # print(
        #     "\n.......RESULT.....\n\nThis image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        # )
        return self.class_names[np.argmax(score)]


#testing
# path = r"C:\Users\user\Desktop\python_ML\Human Emotion Detecor\testing_img\cry.jpg"
# EDE = EmotionDetectEnginee(image_path=path)
# EDE.prediction()