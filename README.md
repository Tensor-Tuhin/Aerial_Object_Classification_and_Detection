# Aerial_Object_Classification_and_Detection
Classification and detection of birds and drones using CNN, Transfer Learning and YOLO.

This project focuses on identifying aerial objects, specifically birds and drones, using deep learning techniques. It combines both image classification and object detection approaches to explore how different models perform on the same problem.

Models Used:

- Convolutional Neural Network (CNN)
- Transfer Learning (MobileNetV2)
- YOLOv8 for object detection

Running the Application:

1. Clone the repository to your local system.
2. Install the required dependencies:

   pip install -r requirements.txt
3. Make sure the model files are placed correctly:
   - Transfer Learning model: "models/tl_model_best.h5"
   - YOLO model: "models/weights/best.pt"
   
   The application expects the "models" folder to be in the same directory as the main application file (app.py). This applies for the other files as well.
4. Start the Streamlit app:
   
   streamlit run app.py

Notes:

- The CNN model is not included due to file size limitations.
- The dataset is also not included due to file size limitations.
- Only the transfer learning and yolo models have been provided for inference.

Conclusion:

This project highlights how different deep learning approaches can be applied to the same task. While classification models perform well, YOLOv8 stands out as the most practical solution because it can both detect and localize objects in an image.
