# Sentiment Analysis App

This project is a sentiment analysis application that utilizes a Flask backend for processing images and predicting sentiment based on the extracted text. The frontend is built using Streamlit, providing an interactive user interface for users to upload images and view predictions.

## Project Structure

```
sentiment-analysis-app
├── backend
│   ├── flask_app.py          # Flask application for image processing and sentiment prediction
│   ├── best_model.keras      # Saved Keras model for sentiment analysis
│   ├── vectorizer_vocab.pkl  # Vocabulary for text vectorization
│   └── requirements.txt      # Python dependencies for the backend
├── frontend
│   ├── streamlit_app.py      # Streamlit application for user interface
│   └── requirements.txt      # Python dependencies for the frontend
└── README.md                 # Project documentation
```

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd sentiment-analysis-app
   ```

2. **Set up the backend:**
   - Navigate to the `backend` directory:
     ```
     cd backend
     ```
   - Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```

3. **Set up the frontend:**
   - Navigate to the `frontend` directory:
     ```
     cd ../frontend
     ```
   - Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```

## Usage

### Running the Backend

1. Navigate to the `backend` directory:
   ```
   cd backend
   ```

2. Start the Flask application:
   ```
   python flask_app.py
   ```

### Running the Frontend

1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```

2. Start the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```

## Features

- Upload images for sentiment analysis.
- Extract text from images using OCR.
- Predict sentiment categories: humour, sarcasm, offensive, and motivational.
- Display extracted text and prediction results in the Streamlit interface.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.