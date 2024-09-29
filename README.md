# Face Verification with Siamese Network using Kivy and TensorFlow

This project implements a face verification system using a Siamese Neural Network in TensorFlow, integrated with a Kivy-based graphical user interface (GUI) for real-time face verification via a webcam. The system allows you to capture and verify images using a pre-trained model.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model](#model)
- [Verification Process](#verification-process)
- [Real-Time Webcam Verification](#real-time-webcam-verification)
- [License](#license)

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-repo/face-verification.git
cd face-verification
```
### Set Up Environment

It’s recommended to create a virtual environment to manage dependencies:

```bash
Set Up Environment

It’s recommended to create a virtual environment to manage dependencies:
```
### Install Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```
If you don’t have requirements.txt, manually install the following libraries:

```bash
pip install kivy tensorflow opencv-python numpy
```
## Dependencies

- Kivy: For building the GUI.
- TensorFlow: For loading the pre-trained Siamese model.
- OpenCV: For capturing webcam footage.
- NumPy: For data processing.

Ensure that your environment also supports GPU usage for TensorFlow, especially if you’re dealing with a large model.

### Project Structure

```bash
.
├── application_data          # Directory containing images for verification
│   ├── input_image           # Directory to store the captured image from the webcam
│   └── verification_images   # Directory with verification images
├── layers.py                 # Custom TensorFlow L1 distance layer
├── face_id.py                # Main Kivy App for face verification
├── siamesemodel.h5           # Pre-trained Siamese model
└── README.md                 # Documentation
```
- layers.py: Contains the custom L1 distance layer used by the Siamese network.
- face_id.py: Implements the face verification system with Kivy for the GUI.
- siamesemodel.h5: Pre-trained Siamese model used for face verification.

### Usage

1. Make sure your webcam is connected.
2. Run the application:

```bash
python face_id.py
```

3. The GUI will open, showing the live feed from your webcam.
4. Capture the image by pressing the Verify button.

## Model

The face verification system uses a Siamese Neural Network with a custom L1 distance layer (layers.py) to compare two face embeddings and determine their similarity. The pre-trained model (siamesemodel.h5) was trained on face image pairs.

### L1 Distance Layer

The custom L1 distance layer is defined in layers.py:

```bash
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
```
This layer calculates the absolute difference between two face embeddings to measure their similarity.

## Verification Process

1. Capture Image: The system captures an image from the webcam feed and stores it in the application_data/input_image directory.
2. Preprocess Image: The image is resized and normalized before being passed to the model.
3. Model Prediction: The pre-trained model compares the captured image with the images stored in the application_data/verification_images directory.
4. Verification: If the similarity score exceeds a predefined threshold, the system will mark the person as “Verified.”

The verification function is in the face_id.py file:

```bash
def verify(self, *args):
    ...
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

        # Prediction using the Siamese model
        result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    ...
```

## Real-Time Webcam Verification

The app continuously streams video from the webcam, and on pressing the Verify button, it captures a snapshot and runs the verification process. The result is shown in the GUI, and the status (Verified/Unverified) is updated on the screen.

### Kivy Layout

The layout consists of:

- Webcam Feed: Displays the live feed from the webcam.
- Verify Button: Captures the image from the webcam and runs the verification process.
- Verification Label: Shows whether the person is verified or not.

```bash
def build(self):
    # Button, Image, Text Comp
    Window.clearcolor = (0.1, 0.1, 0.1, 1)

    self.web_cam = Image(size_hint=(1, 0.75))
    self.button = Button(
        text="Verify",
        on_press=self.verify,
        size_hint=(1, 0.1),
        font_size='20sp',
        background_color=(0.2, 0.6, 0.8, 1),
        color=(1, 1, 1, 1),
        bold=True
    )
    self.verification_label = Label(
        text="Verification Uninitiated",
        size_hint=(1, 0.1),
        font_size='18sp',
        color=(1, 1, 1, 1),
        halign='center',
        valign='middle'
    )
    layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
    layout.add_widget(self.web_cam)
    layout.add_widget(self.button)
    layout.add_widget(self.verification_label)
    return layout
```
## License

This project is licensed under the MIT License. See the LICENSE file for more information.

```bash

### Key Points in the README:

1. **Project Overview**: Brief explanation of what the project does.
2. **Installation**: How to set up and run the project.
3. **Dependencies**: Clear indication of what libraries and tools are needed.
4. **Project Structure**: Outline of the project directory.
5. **Usage**: Instructions on how to use the app.
6. **Model Description**: Brief on how the Siamese model works with the custom layer.
7. **Verification Process**: Explanation of the image verification process.
8. **Real-Time Verification**: How the GUI integrates with real-time webcam feed and verification.
9. **License**: Mention of the project’s license. 

Let me know if you need any further adjustments!
```


