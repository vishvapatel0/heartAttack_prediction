
# Heart Attack Prediction Model

## Overview
This project focuses on predicting the likelihood of a heart attack using a Random Forest Classifier. The dataset consists of various health parameters, including blood pressure, cholesterol levels, and other relevant medical and lifestyle factors. The model is trained using hyperparameter tuning and validated using cross-validation techniques.

## Dataset
The dataset used for training and testing the model is stored in a CSV file. It contains both numerical and categorical features. Key preprocessing steps include:
- Handling missing values
- Encoding categorical variables using Label Encoding
- Standardizing numerical features using StandardScaler
- Splitting data into training and testing sets

## Model Details
The project employs a **Random Forest Classifier**, which is optimized using **GridSearchCV**. The model is evaluated based on accuracy, classification report, and feature importance analysis. Additionally, a learning curve is plotted to analyze bias and variance trade-offs.

## Key Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and normalizing numerical features.
- **Hyperparameter Tuning**: Optimizing the Random Forest model using GridSearchCV.
- **Model Evaluation**: Analyzing accuracy, classification reports, feature importance, and learning curves.
- **File Saving**:
  - The trained model is saved as `random_forest_heart_attack.pkl`.
  - The scaler used for normalization is saved as `scaler.pkl`.
  - Label encoders for categorical variables are saved as `label_encoders.pkl`.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

### Running the Model
1. Load the dataset.
2. Preprocess the data.
3. Train the Random Forest Classifier.
4. Evaluate the model.
5. Save the trained model, scaler, and label encoders for future use.

Run the script using:
```bash
python heart_attack_prediction.py
```

## Results
- **Optimized Model Accuracy**: Displayed in the console.
- **Feature Importance**: Visualized using bar plots.
- **Learning Curve Analysis**: Determines whether the model is overfitting or underfitting.

## Future Improvements
- Exploring deep learning techniques for better performance.
- Deploying the model using a web application.
- Incorporating real-time data for continuous model improvement.

## Contributors
- **Vishva Patel**
- Other contributors are welcome to collaborate!

## License
This project is licensed under the MIT License.
