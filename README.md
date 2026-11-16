\# Real-Time Sign Language Recognition System



A computer vision and machine learning-based application that recognizes hand gestures in real time using a webcam and converts them into text.



\## Project Overview



The Real-Time Sign Language Recognition System is designed to improve communication accessibility by recognizing hand gestures through a webcam.



It uses MediaPipe to extract hand landmarks and a machine learning classifier to identify corresponding sign language gestures.



\## Key Features



\- Real-time hand gesture recognition

\- Webcam-based detection

\- Hand landmark extraction using MediaPipe

\- Machine learning-based gesture classification

\- Support for 24 gesture categories

\- Flask-based web application

\- Custom dataset for model training



\## Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| OpenCV | Image processing |

| MediaPipe | Hand landmark detection |

| Scikit-learn | Machine learning |

| Flask | Web application |

| Flask-CORS | Cross-origin resource sharing |

| Matplotlib | Data visualization |



\## Project Structure



```text

Realtime-Sign-LanguageRecognition/

│

├── app.py

├── train\_classifier.py

├── create\_dataset.py

├── inference\_classifier.py

├── model.p

├── data.pickle

├── requirements.txt

│

├── data/

├── static/

├── templates/

│

└── README.md

```



\## Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/pragatheashwaran/Realtime-Sign-LanguageRecognition.git

```



\### 2. Navigate to the Project Directory



```bash

cd Realtime-Sign-LanguageRecognition

```



\### 3. Create a Virtual Environment



```bash

python -m venv venv

```



\### 4. Activate the Virtual Environment



For Windows:



```bash

venv\\Scripts\\activate

```



\### 5. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 6. Run the Application



```bash

python app.py

```



Open the local URL displayed in the terminal in your browser.



\## Machine Learning Workflow



1\. Collect hand gesture images.

2\. Extract hand landmarks using MediaPipe.

3\. Prepare the dataset.

4\. Train the machine learning classifier.

5\. Save the trained model.

6\. Perform real-time gesture predictions.



\## Dataset



The project contains a custom dataset with approximately 2,200 images distributed across 24 gesture categories.



\## Project Objective



To develop an accessible communication solution using computer vision and machine learning for real-time hand gesture recognition.



