This project showcases a cutting-edge Face Mask Recognition System that combines the power of advanced deep learning models with traditional machine learning techniques. It is designed to identify whether individuals are wearing masks properly, improperly, or not at all. The system supports real-time video analysis and offers robust performance even in challenging conditions.

🚀 Key Features
Real-Time Mask Detection:

Utilizes RetinaFace for accurate face detection in live video feeds or pre-recorded footage.
Classifies faces into three categories:
No Mask
Properly Worn Mask
Improperly Worn Mask
Displays results with bounding boxes and clear annotations on detected faces.
High-Performance CNN Model:

Implements a deep Convolutional Neural Network with layers optimized for feature extraction.
Employs techniques like batch normalization, LeakyReLU activation, and dropout regularization for improved generalization.
Trained using data augmentation and validated on diverse datasets for real-world applicability.
Achieves impressive accuracy (over 90%) on test datasets.
HOG + Machine Learning:

Applies Histogram of Oriented Gradients (HOG) for feature extraction.
Incorporates SVM (Support Vector Machine) and MLP (Multi-Layer Perceptron) classifiers as alternative detection methods.
Demonstrates robust performance with detailed classification metrics.
Class Balancing with SMOTE:

Overcomes dataset imbalance using the SMOTE (Synthetic Minority Over-sampling Technique) to improve model fairness and accuracy across all classes.
Visualization & Reporting:

Generates performance metrics, including confusion matrices, classification reports, and ROC curves.
Visualizes class distributions and detection results using intuitive plots.
⚙️ Technologies and Frameworks
Deep Learning: TensorFlow, Keras
Machine Learning: Scikit-learn, Imbalanced-learn
Computer Vision: OpenCV, Scikit-image
Real-Time Processing: RetinaFace, OpenCV
Visualization: Matplotlib, Seaborn
🎯 Applications
Ensuring compliance with public health guidelines in crowded areas.
Monitoring mask usage in real-time for workplaces, events, or transportation hubs.
Deploying in security systems to enhance safety measures during pandemics.
📊 Performance Highlights
CNN Model: Achieved over 90% accuracy on multi-class classification tasks.
HOG + SVM/MLP Models: Provided reliable alternative methods for classification.
Real-Time Inference: Successfully detects and classifies faces in videos with minimal latency.
🔧 How It Works
Preprocessing:
Organizes the dataset into training and testing splits.
Balances class distributions to ensure model fairness.
Training:
Builds and trains the CNN using augmented datasets.
Experiments with HOG features and SVM/MLP classifiers for comparative analysis.
Inference:
Detects faces in video streams.
Classifies each face into one of the predefined categories.
Annotates the video output with bounding boxes and labels.
