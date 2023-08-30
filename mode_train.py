import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model



class EmotionDetect():
    def __init__(self):
        self.data_dir = pathlib.Path(r'data\train')
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        self.epoch = 20
        self.model_save_path = 'models/'
    
    def get_training_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size= self.batch_size)
        return train_ds

    def get_valid_data(self):
        val_ds = tf.keras.utils.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size= self.batch_size)
        return val_ds
    
    def get_class_name(self,train_ds):
        class_names = train_ds.class_names
        return class_names
    
    def show_image(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.val_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")


    def get_model(self,class_names):
        num_classes = len(class_names)
        model = Sequential([
        layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print(model.summary())
        return model

    def training(self,model,train_ds,val_ds):
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=self.epoch
        )
        return model,history
    
    def get_accuracy(self,history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epoch)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def executer(self):
        #get dataset
        train_ds = self.get_training_data()
        valid_ds = self.get_valid_data()
        class_names = self.get_class_name(train_ds=train_ds)
        #Auto tunning
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model = self.get_model(class_names=class_names)

        trained_model,history = self.training(model=model,train_ds=train_ds, val_ds=val_ds)
        #save model
        trained_model.save(self.model_save_path +'imageclassifier.h5')
        print('Model saved successfully.......!!!!!!!!!!!')
        #find accuracy
        self.get_accuracy(history= history)



#testing
emo_det = EmotionDetect()
emo_det.executer()