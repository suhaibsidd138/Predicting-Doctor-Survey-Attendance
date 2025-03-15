import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request, send_file, jsonify

# ----- Data Processing and Model Training Functions -----

def load_and_prepare_data(file_path):
    """
    Load and prepare the NPI dataset for modeling.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['NPI', 'Login Time', 'Logout Time'])
    
    # Feature engineering
    # Extract time features from Login Time
    df['Login Time'] = pd.to_datetime(df['Login Time'])
    df['Logout Time'] = pd.to_datetime(df['Logout Time'])
    
    df['Login Hour'] = df['Login Time'].dt.hour
    df['Login Minute'] = df['Login Time'].dt.minute
    df['Login Day'] = df['Login Time'].dt.dayofweek
    
    # Calculate engagement duration in minutes
    df['Engagement Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60
    
    # Create a target variable (example: high engagement could be a proxy for survey participation)
    # You should adjust this based on your actual data and domain knowledge
    df['Likely_To_Participate'] = (df['Time spent'] > df['Time spent'].median()) & (df['Count of Attempts'] > 1)
    
    return df

def build_model(df):
    """
    Build a machine learning model to predict survey participation.
    """
    # Define features and target
    X = df[['Login Hour', 'Login Minute', 'Login Day', 'Engagement Duration', 'Time spent', 'Count of Attempts']]
    
    # Add categorical features if available
    if 'Speciality' in df.columns:
        X = pd.concat([X, pd.get_dummies(df['Speciality'], prefix='Spec')], axis=1)
    
    if 'Region' in df.columns:
        X = pd.concat([X, pd.get_dummies(df['Region'], prefix='Region')], axis=1)
    
    y = df['Likely_To_Participate']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model

def save_model(model, file_path='npi_prediction_model.pkl'):
    """
    Save the trained model to a file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def predict_participants(df, model, input_time):
    """
    Predict which doctors are most likely to participate at the given time.
    
    Args:
        df: DataFrame with doctor data
        model: Trained prediction model
        input_time: String in format "HH:MM" (24-hour format)
    
    Returns:
        DataFrame with NPIs of doctors likely to participate
    """
    # Parse the input time
    hour, minute = map(int, input_time.split(':'))
    
    # Create a copy of the dataframe with unique NPIs
    unique_doctors = df.drop_duplicates(subset='NPI')
    
    # Prepare features for prediction
    X_pred = unique_doctors[['Login Hour', 'Login Minute', 'Login Day', 'Engagement Duration', 'Time spent', 'Count of Attempts']]
    
    # Add categorical features if used in training
    if 'Speciality' in df.columns and any(col.startswith('Spec_') for col in model.feature_names_in_):
        X_pred = pd.concat([X_pred, pd.get_dummies(unique_doctors['Speciality'], prefix='Spec')], axis=1)
    
    if 'Region' in df.columns and any(col.startswith('Region_') for col in model.feature_names_in_):
        X_pred = pd.concat([X_pred, pd.get_dummies(unique_doctors['Region'], prefix='Region')], axis=1)
    
    # Ensure X_pred has all the columns the model expects
    missing_cols = set(model.feature_names_in_) - set(X_pred.columns)
    for col in missing_cols:
        X_pred[col] = 0
    
    # Ensure columns are in the same order as during training
    X_pred = X_pred[model.feature_names_in_]
    
    # Get participation probabilities
    participation_probs = model.predict_proba(X_pred)[:, 1]  # Probability of class 1 (likely to participate)
    
    # Add probabilities to the doctors dataframe
    unique_doctors['Participation_Probability'] = participation_probs
    
    # Filter doctors with high probability of participation at the given time
    # We can adjust the time difference tolerance
    time_diff = abs(unique_doctors['Login Hour'] - hour) + abs(unique_doctors['Login Minute'] - minute) / 60
    unique_doctors['Time_Relevance'] = 1 / (1 + time_diff)  # Higher value for closer times
    
    # Combined score: combination of participation probability and time relevance
    unique_doctors['Combined_Score'] = unique_doctors['Participation_Probability'] * unique_doctors['Time_Relevance']
    
    # Sort by combined score and get top doctors
    result = unique_doctors.sort_values('Combined_Score', ascending=False)
    
    # Return the NPIs and relevant information
    return result[['NPI', 'Speciality', 'Region', 'Participation_Probability', 'Combined_Score']]

# ----- Flask Web Application -----

app = Flask(__name__)

# Model and data paths
MODEL_PATH = 'npi_prediction_model.pkl'
DATA_PATH = 'npi_data.csv'  # Replace with your actual data path

# Create the templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Doctor Survey Participation Predictor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 800px;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            margin-bottom: 30px;
            text-align: center;
        }
        .form-container {
            margin-bottom: 30px;
        }
        .result-container {
            margin-top: 30px;
            display: none;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .preview-table {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Doctor Survey Participation Predictor</h1>
            <p class="lead">Predict which doctors are most likely to participate in your survey based on their historical data</p>
        </div>
        
        <div class="form-container">
            <form id="prediction-form">
                <div class="mb-3">
                    <label for="timeInput" class="form-label">Enter Time (24-hour format)</label>
                    <input type="time" class="form-control" id="timeInput" name="time" required>
                    <div class="form-text">Example: 14:30 for 2:30 PM</div>
                </div>
                <button type="submit" class="btn btn-primary">Predict Participation</button>
            </form>
        </div>
        
        <div class="loader" id="loader"></div>
        
        <div class="result-container" id="result-container">
            <h3>Results Preview</h3>
            <p>Below are the top 5 doctors most likely to participate in your survey at the specified time:</p>
            
            <div class="table-responsive preview-table">
                <table class="table table-striped" id="preview-table">
                    <thead>
                        <tr>
                            <th>NPI</th>
                            <th>Speciality</th>
                            <th>Region</th>
                            <th>Probability</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody id="preview-body">
                        <!-- Preview data will be inserted here -->
                    </tbody>
                </table>
            </div>
            
            <div class="mt-4">
                <a href="#" id="download-link" class="btn btn-success">Download Full Results (CSV)</a>
                <button class="btn btn-secondary ms-2" id="new-prediction-btn">New Prediction</button>
            </div>
        </div>
        
        <div class="alert alert-danger mt-4" id="error-message" style="display: none;">
            <!-- Error messages will appear here -->
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('prediction-form');
            const loader = document.getElementById('loader');
            const resultContainer = document.getElementById('result-container');
            const errorMessage = document.getElementById('error-message');
            const previewBody = document.getElementById('preview-body');
            const downloadLink = document.getElementById('download-link');
            const newPredictionBtn = document.getElementById('new-prediction-btn');
            
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loader, hide results and errors
                loader.style.display = 'block';
                resultContainer.style.display = 'none';
                errorMessage.style.display = 'none';
                
                // Get form data
                const formData = new FormData(form);
                
                // Send AJAX request
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loader.style.display = 'none';
                    
                    if (data.error) {
                        // Show error message
                        errorMessage.textContent = data.error;
                        errorMessage.style.display = 'block';
                    } else {
                        // Update preview table
                        previewBody.innerHTML = '';
                        data.preview.forEach(function(doctor) {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${doctor.NPI}</td>
                                <td>${doctor.Speciality || 'N/A'}</td>
                                <td>${doctor.Region || 'N/A'}</td>
                                <td>${(doctor.Participation_Probability * 100).toFixed(2)}%</td>
                                <td>${doctor.Combined_Score.toFixed(3)}</td>
                            `;
                            previewBody.appendChild(row);
                        });
                        
                        // Update download link
                        downloadLink.href = `/download/${data.file_name}`;
                        
                        // Show results
                        resultContainer.style.display = 'block';
                    }
                })
                .catch(error => {
                    loader.style.display = 'none';
                    errorMessage.textContent = 'An error occurred. Please try again.';
                    errorMessage.style.display = 'block';
                    console.error('Error:', error);
                });
            });
            
            // New prediction button
            newPredictionBtn.addEventListener('click', function() {
                form.reset();
                resultContainer.style.display = 'none';
            });
        });
    </script>
</body>
</html>''')

def load_model_and_data():
    # Load the model
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return None, None, "Model file not found. Please train the model first."
    
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Recreate necessary time features
        df['Login Time'] = pd.to_datetime(df['Login Time'])
        df['Logout Time'] = pd.to_datetime(df['Logout Time'])
        
        df['Login Hour'] = df['Login Time'].dt.hour
        df['Login Minute'] = df['Login Time'].dt.minute
        df['Login Day'] = df['Login Time'].dt.dayofweek
        
        # Calculate engagement duration in minutes
        df['Engagement Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60
        
    except FileNotFoundError:
        return model, None, "Data file not found."
    except Exception as e:
        return model, None, f"Error loading data: {str(e)}"
    
    return model, df, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input time from the request
    input_time = request.form.get('time')
    
    # Validate time format
    try:
        datetime.strptime(input_time, '%H:%M')
    except ValueError:
        return jsonify({'error': 'Invalid time format. Please use HH:MM (24-hour format).'})
    
    # Load model and data
    model, df, error = load_model_and_data()
    
    if error:
        return jsonify({'error': error})
    
    # Make prediction
    try:
        result_df = predict_participants(df, model, input_time)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        
        # Save to a file for download
        file_name = f"likely_participants_at_{input_time.replace(':', '')}.csv"
        result_df.to_csv(file_name, index=False)
        
        return jsonify({
            'success': True,
