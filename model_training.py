import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data_preprocessing import DataPreprocessor
import xgboost as xgb
import logging
import sys
import os
from typing import Tuple, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ModelTrainer:
    def __init__(self):
        """Initialize the ModelTrainer with error handling."""
        try:
            self.model = None
            self.preprocessor = DataPreprocessor()
            logging.info("ModelTrainer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ModelTrainer: {str(e)}")
            raise
    
    def validate_data(self, data: Any) -> bool:
        """
        Validate the input data before training.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            if data is None:
                logging.error("Data is None")
                return False
            
            required_columns = [
                'LotArea', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 
                'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
                'OverallQual', 'OverallCond', 'Neighborhood', 'SalePrice'
            ]
            
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                return False
            
            if data.empty:
                logging.error("Data is empty")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {str(e)}")
            return False
    
    def train_model(self, data_path: str, model_type: str = 'random_forest') -> bool:
        """
        Train the model with enhanced error handling and logging.
        
        Args:
            data_path: Path to the training data
            model_type: Type of model to train ('random_forest' or 'xgboost')
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Load and preprocess data
            logging.info(f"Loading data from {data_path}")
            data = self.preprocessor.load_data(data_path)
            
            if not self.validate_data(data):
                return False
            
            logging.info("Preprocessing data")
            try:
                X, y = self.preprocessor.preprocess_data(data)
            except Exception as e:
                logging.error(f"Data preprocessing failed: {str(e)}")
                return False
            
            # Split the data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                logging.info("Data split successfully")
            except Exception as e:
                logging.error(f"Data splitting failed: {str(e)}")
                return False
            
            # Initialize and train the model
            logging.info(f"Training {model_type} model")
            try:
                if model_type == 'random_forest':
                    self.model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == 'xgboost':
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    logging.error(f"Unsupported model type: {model_type}")
                    return False
                
                self.model.fit(X_train, y_train)
                logging.info("Model training completed")
                
            except Exception as e:
                logging.error(f"Model training failed: {str(e)}")
                return False
            
            # Evaluate the model
            try:
                self._evaluate_model(X_train, X_test, y_train, y_test, X, y)
            except Exception as e:
                logging.error(f"Model evaluation failed: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"An unexpected error occurred during training: {str(e)}")
            return False
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test, X, y) -> None:
        """
        Evaluate the model and log performance metrics.
        """
        try:
            # Calculate scores
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            # Log evaluation metrics
            logging.info("\nModel Evaluation Metrics:")
            logging.info(f"Training R² Score: {train_score:.4f}")
            logging.info(f"Testing R² Score: {test_score:.4f}")
            logging.info(f"Root Mean Squared Error: ${rmse:,.2f}")
            logging.info(f"R² Score: {r2:.4f}")
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Average CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def save_model(self, model_path: str = 'house_price_model.joblib') -> bool:
        """
        Save the trained model with error handling.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            if self.model is None:
                logging.error("No model to save. Please train the model first.")
                return False
            
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
            
            joblib.dump(model_data, model_path)
            logging.info(f"Model and preprocessor saved to {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(model_path: str = 'house_price_model.joblib') -> Optional['ModelTrainer']:
        """
        Load the trained model with error handling.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            ModelTrainer or None: Loaded model trainer instance or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                return None
                
            model_data = joblib.load(model_path)
            trainer = ModelTrainer()
            trainer.model = model_data['model']
            trainer.preprocessor = model_data['preprocessor']
            logging.info(f"Model loaded successfully from {model_path}")
            return trainer
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None

def main():
    """Main function to train and save the model."""
    try:
        trainer = ModelTrainer()
        
        # Check if data file exists
        data_path = "house_data.csv"
        if not os.path.exists(data_path):
            logging.error(f"Training data not found: {data_path}")
            print(f"\nError: Training data file '{data_path}' not found.")
            print("Solution:")
            print("1. Download the Kaggle House Prices dataset")
            print("2. Save it as 'house_data.csv' in the project directory")
            print("3. Run this script again")
            return
        
        # Train the model
        success = trainer.train_model(data_path, model_type='xgboost')
        if success:
            # Save the model
            if trainer.save_model():
                logging.info("Model training and saving completed successfully")
            else:
                logging.error("Failed to save the model")
        else:
            logging.error("Model training failed")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        print("\nError: An unexpected error occurred during model training.")
        print("Please check the log file 'model_training.log' for details.")
        print("\nPossible solutions:")
        print("1. Verify that all required packages are installed")
        print("2. Check that the input data is correctly formatted")
        print("3. Ensure sufficient system memory is available")
        print("4. Review the error message in the log file")

if __name__ == "__main__":
    main() 