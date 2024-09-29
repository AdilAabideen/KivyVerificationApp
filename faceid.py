# Import dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import UX components

from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

# Import kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#Building the FUll Stack
class CamApp(App):

    def build(self):
        #Button , Image, Text Comp
        Window.clearcolor = (0.1, 0.1, 0.1, 1)

        # Webcam feed (Image)
        self.web_cam = Image(size_hint=(1, 0.75))

        # Verification Button
        self.button = Button(
            text="Verify",
            on_press=self.verify,
            size_hint=(1, 0.1),
            font_size='20sp',
            background_color=(0.2, 0.6, 0.8, 1),  # Custom background color
            color=(1, 1, 1, 1),  # White text color
            bold=True
        )

        # Verification Status Label
        self.verification_label = Label(
            text="Verification Uninitiated",
            size_hint=(1, 0.1),
            font_size='18sp',
            color=(1, 1, 1, 1),  # White text
            halign='center',
            valign='middle'
        )

        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Adding the Components to Layout
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Loading Siamese Model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # Setting my Video Capture (Webcam)
        self.capture = cv2.VideoCapture(0)

        # Schedule to keep updating the webcam feed
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    def update(self, *args):


        # Read frame from opencv
        ret, frame = self.capture.read()
        # Cut down frame
        frame = frame[500:1300, 500:1300, :]
        frame = cv2.flip(frame, 1)  # Flip horizontally and vertically

        # Flip horizontall 
        buf = cv2.flip(frame, 0).tostring()

        #Convert image to texture
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


    # PRe PRocess function from Jupyter Notebook
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (105,105))
        # Scale image to be between 0 and 1 
        img = img / 255.0

        # Return image
        return img


    # Verification Function From Jupyter Notebook
    def verify(self, *args):

        #Thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Getting Image from Webcam and Saving
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[500:1300, 500:1300, :]
        cv2.imwrite(SAVE_PATH, frame)
        cv2.imwrite(SAVE_PATH, frame)
       

        results= []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Makingf the Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Thresholds are the metric above which a prediciton is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        # Verification Thresholds are the proportion of + prediction / total + samples

        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        #Setting out texts
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details to improve Decision Metrics
         # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        #Logger.info(detection)
        #Logger.info(verification)
        #Logger.info(verified)


        return results, verified

if __name__ == '__main__':
    CamApp().run()