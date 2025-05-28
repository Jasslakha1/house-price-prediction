import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load the dataset from a CSV file."""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Preprocess the data by handling missing values, encoding categorical variables, and scaling."""
        # Select relevant features
        selected_features = [
            'LotArea', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 
            'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
            'OverallQual', 'OverallCond', 'Neighborhood'
        ]
        target = 'SalePrice'
        
        # Keep only selected features and target
        data = data[selected_features + [target]].copy()
        
        # Separate numerical and categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(exclude=['object']).columns.difference([target])
        
        # Store feature columns for prediction
        self.feature_columns = list(numerical_cols) + list(categorical_cols)
        
        # Handle missing values
        data[numerical_cols] = self.numerical_imputer.fit_transform(data[numerical_cols])
        
        if len(categorical_cols) > 0:
            data[categorical_cols] = self.categorical_imputer.fit_transform(data[categorical_cols])
            
            # Encode categorical variables
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        # Scale numerical features
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data[target]
        
        return X, y
    
    def preprocess_input(self, input_data):
        """Preprocess a single input or batch of inputs for prediction."""
        input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Handle categorical features
        categorical_cols = [col for col in self.feature_columns if col in self.label_encoders]
        for col in categorical_cols:
            input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
        
        # Handle numerical features
        numerical_cols = [col for col in self.feature_columns if col not in self.label_encoders]
        input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])
        
        return input_df[self.feature_columns] 