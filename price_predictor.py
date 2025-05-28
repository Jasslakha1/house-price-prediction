from model_training import ModelTrainer

class HousePricePredictor:
    def __init__(self, model_path='house_price_model.joblib'):
        """Initialize the predictor by loading the trained model."""
        self.trainer = ModelTrainer.load_model(model_path)
        if self.trainer is None:
            raise ValueError("Failed to load the model")
    
    def predict_price(self, features):
        """
        Predict house price based on input features.
        
        Args:
            features (dict): A dictionary containing house features:
                - LotArea: Lot size in square feet
                - YearBuilt: Original construction date
                - BedroomAbvGr: Number of bedrooms above ground
                - FullBath: Number of full bathrooms
                - HalfBath: Number of half bathrooms
                - TotRmsAbvGrd: Total rooms above ground (excluding bathrooms)
                - GarageCars: Size of garage in car capacity
                - GarageArea: Size of garage in square feet
                - OverallQual: Overall material and finish quality (1-10)
                - OverallCond: Overall condition rating (1-10)
                - Neighborhood: Physical location within the city
        
        Returns:
            float: Predicted house price in dollars
        """
        try:
            # Preprocess the input features
            X = self.trainer.preprocessor.preprocess_input(features)
            
            # Make prediction
            predicted_price = self.trainer.model.predict(X)[0]
            
            return max(0, predicted_price)  # Ensure non-negative price
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_feature_requirements(self):
        """Return the list of required features and their descriptions."""
        return {
            'LotArea': 'Lot size in square feet (numeric)',
            'YearBuilt': 'Original construction date (year)',
            'BedroomAbvGr': 'Number of bedrooms above ground (numeric)',
            'FullBath': 'Number of full bathrooms (numeric)',
            'HalfBath': 'Number of half bathrooms (numeric)',
            'TotRmsAbvGrd': 'Total rooms above ground excluding bathrooms (numeric)',
            'GarageCars': 'Size of garage in car capacity (numeric)',
            'GarageArea': 'Size of garage in square feet (numeric)',
            'OverallQual': 'Overall material and finish quality (1-10)',
            'OverallCond': 'Overall condition rating (1-10)',
            'Neighborhood': 'Physical location within the city (text)'
        }

if __name__ == "__main__":
    # Example usage
    try:
        predictor = HousePricePredictor()
        
        # Example features for a house
        sample_features = {
            'LotArea': 8450,
            'YearBuilt': 2000,
            'BedroomAbvGr': 3,
            'FullBath': 2,
            'HalfBath': 1,
            'TotRmsAbvGrd': 8,
            'GarageCars': 2,
            'GarageArea': 400,
            'OverallQual': 7,
            'OverallCond': 6,
            'Neighborhood': 'NAmes'
        }
        
        # Make prediction
        predicted_price = predictor.predict_price(sample_features)
        
        if predicted_price is not None:
            print(f"\nPredicted House Price: ${predicted_price:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}") 