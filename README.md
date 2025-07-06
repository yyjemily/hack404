# Dentura: The Dental AI Assistant

Dentura is a full-stack project that uses a FastAPI backend, a Node.js, and a React frontend to perform
real-time X-ray image analysis for teeth. A pretrained Densenet model was used to train and fine tune
Dentura's ML model on a provided dataset using 20 epochs in a Google Colab enviornment. Credits to Mohamadreza 
Momeni for providing this dataset in https://www.kaggle.com/datasets/imtkaggleteam/dental-opg-xray-dataset.
Further credits to Manar Abu Talib, Mohammad Adel Moufti, Qassim Nasir, Yousuf Kabbani, Dana Aljaghber, Yaman Afadar,
Transfer Learning-Based Classifier to Automate the Extraction of False X-Ray Images From Hospital's Database, 
International Dental Journal, Volume 74, Issue 6, 2024, Pages 1471-1482, ISSN 0020-6539, 
https://doi.org/10.1016/j.identj.2024.08.002. (https://www.sciencedirect.com/science/article/pii/S0020653924014138).
The FastAPI backend was used to handle the best trained model of the dentistry dataset. The API then 
automatically calls an endpoint service built in the Node.js backend. The React webpage frontend further polled the 
backend periodically, to ultimately summarize the prediagnositics of a patient's teeth to the dentist.

## Features

- **Real-Time X-Ray Image Analysis:**  
  Uses a pretrained Densenet model to analyze teeth conditions.

- **Diagnosis Summaries:**  
  Summarizes a patient's teeth conditions to aid dentists in prediagnosis.

- **Dynamic Frontend:**  
  A React webpage that polls the latest X-Ray image periodicially, hosts a chatbot for
  patient diagnosis, and overlays bounding boxes or alert symbols based on the danger level.

- **User Interaction:**
A toggle button allows the user to upload X-Ray images for patient diagnosis.

## Project Setup

### Prerequisites
- Python 3+
- Node.js

### Installation
1. JS frontend:
   - Navigate to the frontend directory
   ```bash
    cd frontend
    npm install
   ```
   - Run the js front end with
   ```bash
   npm start
   ```
2. Py backend:
  - Create and activate python virtual environment with python version 3.8.x
    ```bash
    pip install requirements.txt
    touch .env
    ```
    In the '.env' file, add:
    ```
    gemini_api_key = YOUR_GEMINI_API_KEY
    ```
    Then
    ```bash
    python api.py
    ```
