# SIMPAC-2024-

Spectral Signature Taxonomy & Analysis Software (SSTAS)
SSTAS is a novel software tool for detecting and classifying asymptomatic plant diseases using Visible Near Infrared Reflectance Spectroscopy (VNIRS) and Deep Learning. It addresses the limitations of traditional visual-based methods by leveraging spectral data to identify diseases like Anthracnose, Powdery Mildew, and Sooty Mold in mango plants, even when symptoms are not visible.
________________________________________
Table of Contents
1.	Introduction
2.	Key Features
3.	Installation
4.	Usage
5.	Dataset
6.	Model Architecture
7.	Results
8.	Future Improvements
9.	Contributing
10.	License
11.	Acknowledgments
________________________________________
Introduction
Plant diseases pose a significant threat to global agriculture, often leading to yield losses and economic instability. Traditional methods relying on visual symptoms are ineffective for asymptomatic diseases. SSTAS uses spectroscopy and deep learning to detect and classify diseases based on spectral signatures, providing a non-invasive and cost-effective solution.
This software was developed as part of research conducted at Charotar University of Science and Technology (CHARUSAT), Gujarat, India. It achieved a validation accuracy of 98.11%, outperforming existing models.
________________________________________
Key Features
•	Spectral Data Analysis: Detects subtle disease-related variations in spectral data (400–1100 nm).
•	Deep Learning Model: A 3-layer neural network for multiclass disease classification.
•	Non-Invasive Monitoring: Enables real-time, cost-effective plant health monitoring.
•	High Accuracy: Achieves 98.11% validation accuracy and 98.25% testing accuracy.
•	Sustainability: Reduces pesticide use and promotes sustainable farming practices.

Installation
To use SSTAS, follow these steps:
1.	Clone the repository:
git clone https://github.com/HetviDesai-14/SIMPAC-2024-.git
cd SIMPAC-2024-
2.	Install dependencies:
  Ensure Python 3.8+ is installed. Then, install the required libraries:
pip install -r requirements.txt

3.	Download the dataset:
The dataset is available on Mendeley Data(https://data.mendeley.com/datasets/3tfjnhmm23/1). Place it in the data/ directory.
Usage


1. Data Preprocessing
Preprocess the spectral data using the provided script:
python scripts/preprocess_data.py
2. Training the Model
Train the deep learning model:
python scripts/train_model.py
3. Testing the Model
Evaluate the model on the test dataset:
python scripts/test_model.py
4. Visualization
Visualize the results:
python scripts/visualize_results.py

Dataset
The dataset consists of spectral data collected from mango leaves at Anand Agriculture University, Gujarat, India. It includes:
•	Healthy leaves: 376,800 spectral signatures.
•	Anthracnose: 251,136 spectral signatures.
•	Powdery Mildew: 126,368 spectral signatures.
•	Sooty Mold: 112,979 spectral signatures.
The dataset is split into training and testing sets (80:20 ratio).
________________________________________
Model Architecture
The deep learning model consists of:
1.	Input Layer: 128 neurons with ReLU activation.
2.	Hidden Layer: 128 neurons with ReLU activation.
3.	Output Layer: 4 neurons (one for each class) with Softmax activation.
   
Hyperparameters
Hyperparameter	Value
Number of Classes	4
Dropout	0.15
Learning Rate	0.001
Batch Size	16
Number of Epochs	35
Activation Functions	ReLU, Softmax
________________________________________
Results
The model achieved the following performance metrics:
•	Training Accuracy: 97.37%
•	Validation Accuracy: 98.11%
•	Testing Accuracy: 98.25%
•	Testing Loss: 0.011
Comparison with Existing Models
Model	Accuracy
SSTAS (Proposed)	98.11%
MobileNet R-CNN (Singh et al., 2020)	70.53%
GAN Architecture (Andrala et al., 2019)	93.67%
CNN (Thotad et al., 2023)	95.00%
Deep CNN (Polder et al., 2019)	88.00%
________________________________________
Future Improvements
1.	Expand Spectral Range: Incorporate sensors for a broader spectral range.
2.	Real-Time Detection: Integrate with mobile platforms for field use.
3.	Advanced Deep Learning Models: Use CNNs and attention-based models for better feature extraction.
4.	IoT Integration: Enable continuous monitoring using IoT devices.
5.	Cloud-Based Processing: Implement cloud computing for large-scale data analysis.
________________________________________
Contributing
We welcome contributions! If you'd like to contribute, please:
1.	Fork the repository.
2.	Create a new branch for your feature or bugfix.
3.	Submit a pull request with a detailed description of your changes.
For more details, see our Contribution Guidelines.
________________________________________
License
This project is licensed under the MIT License. See the LICENSE file for details.
________________________________________
Acknowledgments
•	Anand Agriculture University for providing the dataset.
•	Charotar Space and Technology Center (CHARUSAT) for supporting the research.
•	Contributors: Jayswal Hardik, Hetvi Desai, Hasti Vakani, Mithil Mistry, and Nilesh Dubey.
