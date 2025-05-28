# Real-Time House Price Prediction System

A Python-based system for predicting house prices in real-time using machine learning. The system features a modern GUI interface built with Tkinter and provides both single predictions and batch processing capabilities.

## Features

- **Machine Learning Model**:
  - Uses advanced regression techniques (Random Forest/XGBoost)
  - Handles multiple house features including size, location, and amenities
  - Includes preprocessing for numerical and categorical variables
  - Trained on real estate market data

- **User-Friendly GUI**:
  - Modern, intuitive interface
  - Real-time price predictions
  - Interactive visualization of prediction history
  - Tooltips for input guidance
  - Support for batch predictions via CSV files

- **Key Functionalities**:
  - Single house price predictions
  - Batch predictions from CSV files
  - Visualization of prediction history
  - Input validation
  - Easy-to-use interface

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd house-price-predictor
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Model**:
   ```bash
   python model_training.py
   ```
   This will train the model using the provided dataset and save it for future use.

2. **Running the GUI**:
   ```bash
   python gui.py
   ```
   This will launch the graphical interface for making predictions.

3. **Making Predictions**:
   - Fill in the house features in the input fields
   - Click "Predict Price" to get a single prediction
   - Use "Load CSV" for batch predictions
   - View prediction history in the graph

## Input Features

The system requires the following input features:
- `LotArea`: Lot size in square feet
- `YearBuilt`: Original construction date
- `BedroomAbvGr`: Number of bedrooms above ground
- `FullBath`: Number of full bathrooms
- `HalfBath`: Number of half bathrooms
- `TotRmsAbvGrd`: Total rooms above ground (excluding bathrooms)
- `GarageCars`: Size of garage in car capacity
- `GarageArea`: Size of garage in square feet
- `OverallQual`: Overall material and finish quality (1-10)
- `OverallCond`: Overall condition rating (1-10)
- `Neighborhood`: Physical location within the city

## CSV Format for Batch Predictions

When using the batch prediction feature, ensure your CSV file has the following columns:
```
LotArea,YearBuilt,BedroomAbvGr,FullBath,HalfBath,TotRmsAbvGrd,GarageCars,GarageArea,OverallQual,OverallCond,Neighborhood
8450,2000,3,2,1,8,2,400,7,6,NAmes
...
```

## Project Structure

- `data_preprocessing.py`: Data preprocessing and feature engineering
- `model_training.py`: Model training and evaluation
- `price_predictor.py`: Price prediction functionality
- `gui.py`: Graphical user interface
- `requirements.txt`: Required Python packages
- `README.md`: Project documentation

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- tkinter (included in Python standard library)
- xgboost
- joblib

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 