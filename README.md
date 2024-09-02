# Eye-Tracking with Pupil Detection

This project implements real-time eye-tracking and gaze visualization using computer vision techniques. It detects facial landmarks, locates the eyes, and tracks pupil movement to estimate the user's gaze direction.

This is the result of the TFG - Desarrollo de un sistema de eye tracking mediante técnicas de vision artificial developed in september 2024 by Jorge Ortega Nieto.

I know that the repositoy is a bit of a mess, because you will find so many trys in the diferent files and many things that can be broken only by looking at them, but the important file is the **OpenPupilTracker.py**.

## Features

- Real-time face and eye detection
- Pupil tracking and gaze estimation
- Multithreaded processing for improved performance
- Kalman filter for smooth tracking
- Visualization of gaze direction on a separate window

## Requirements

### Hardware

- Webcam (built-in or external)
- For remote camera usage: A smartphone with DroidCam app installed (both devices must be on the same Wi-Fi network)

### Software

- Python 3.7+
- OpenCV
- dlib
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Jortegan/TFG-eyeTracking.git
   cd eye-tracking-project
   ```

2. Install the required packages:
   ```
   pip install opencv-python dlib numpy
   ```

3. Download the shape predictor file:
   - Download `shape_predictor_68_face_landmarks.dat` from the dlib model repository
   - Place it in the project root directory
   - Its alredy on the repository, but maybe there a newer version that is better (up to you)

## Usage

1. If using a built-in webcam or USB camera:
   - Ensure it's properly connected and recognized by your system

2. If using DroidCam:
   - Install the DroidCam app on your smartphone
   - Connect both your computer and smartphone to the same Wi-Fi network
   - Open the DroidCam app and note the IP address and port number displayed

3. Update the video capture source in the script:
   - For built-in or USB webcam: `cap = cv2.VideoCapture(0)`
   - For DroidCam: `cap = cv2.VideoCapture("http://IP_ADDRESS:PORT/video")` (replace IP_ADDRESS and PORT with the values from DroidCam)

4. Run the script:
   ```
   python OpenPupilTracker.py
   ```

5. Two windows will appear:
   - 'Frame': Shows the video feed with detected facial landmarks
   - 'Gaze Tracking': Displays the estimated gaze direction

6. To exit the program, press 'q' while focused on either window

## Customization

- Adjust the `scale_factor` variable to change the sensitivity of gaze tracking
- Modify the `frame_skip` value to process more or fewer frames (affects performance and smoothness)
- Tweak the Kalman filter parameters for different smoothing effects

## Troubleshooting

- If face detection is unreliable, ensure proper lighting conditions
- For performance issues, try increasing the `frame_skip` value
- If using DroidCam and unable to connect, check that both devices are on the same network and the IP/port are correct

## Acknowledgments

- dlib for providing the facial landmark detection model
- OpenCV community for the computer vision tools and resources
