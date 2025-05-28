"""
Test file demonstrating common errors and solutions in the house price prediction system.
This file shows how to handle various errors that might occur when using the system.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import traceback

class ErrorDemo:
    def __init__(self):
        self.required_features = [
            'LotArea', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 
            'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
            'OverallQual', 'OverallCond', 'Neighborhood'
        ]

    def demonstrate_file_not_found(self, file_path: str) -> None:
        """
        Demonstrates how to handle FileNotFoundError with proper error messages.
        """
        try:
            # Attempt to open a file
            with open(file_path, 'r') as f:
                data = f.read()
        except FileNotFoundError as e:
            print(f"\nError: Could not find the file '{file_path}'")
            print("Solution: ")
            print("1. Check if the file exists in the correct directory")
            print("2. Verify the file path is correct")
            print("3. Ensure you have read permissions for the file")
            print(f"Full error: {str(e)}\n")

    def demonstrate_missing_dependencies(self) -> None:
        """
        Demonstrates how to handle missing Python dependencies.
        """
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'joblib']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print("\nError: Missing required Python packages:")
            print("The following packages need to be installed:")
            for pkg in missing_packages:
                print(f"- {pkg}")
            print("\nSolution:")
            print("Run the following command to install missing packages:")
            print(f"pip install {' '.join(missing_packages)}\n")

    def demonstrate_invalid_input(self, features: Dict[str, Any]) -> None:
        """
        Demonstrates how to handle invalid input data.
        """
        try:
            # Check for missing required features
            missing_features = [f for f in self.required_features if f not in features]
            if missing_features:
                raise ValueError(f"Missing required features: {', '.join(missing_features)}")

            # Check data types
            for feature, value in features.items():
                if feature != 'Neighborhood':  # Neighborhood is categorical
                    try:
                        float_value = float(value)
                        if feature in ['OverallQual', 'OverallCond'] and not (1 <= float_value <= 10):
                            raise ValueError(f"{feature} must be between 1 and 10")
                    except ValueError:
                        raise ValueError(f"{feature} must be a valid number")

        except ValueError as e:
            print("\nError: Invalid input data")
            print(f"Details: {str(e)}")
            print("\nSolution:")
            print("1. Ensure all required features are provided")
            print("2. Check that numerical values are valid numbers")
            print("3. Verify that ratings are between 1 and 10")
            print("4. Make sure the data types match the requirements\n")

    def demonstrate_model_load_error(self, model_path: str) -> None:
        """
        Demonstrates how to handle model loading errors.
        """
        try:
            import joblib
            model = joblib.load(model_path)
        except Exception as e:
            print("\nError: Failed to load the model")
            print(f"Details: {str(e)}")
            print("\nSolution:")
            print("1. Verify the model file exists and is not corrupted")
            print("2. Check if the model was saved with a compatible joblib version")
            print("3. Ensure you have sufficient memory to load the model")
            print("4. Try retraining and saving the model\n")

    def demonstrate_prediction_error(self, features: Dict[str, Any]) -> None:
        """
        Demonstrates how to handle prediction errors.
        """
        try:
            # Simulate prediction
            if not isinstance(features, dict):
                raise TypeError("Features must be provided as a dictionary")

            # Validate feature values
            for feature, value in features.items():
                if feature not in self.required_features:
                    raise ValueError(f"Unknown feature: {feature}")

            # More validation could be added here...

        except Exception as e:
            print("\nError: Prediction failed")
            print(f"Details: {str(e)}")
            print("\nSolution:")
            print("1. Check that all input features are correctly formatted")
            print("2. Verify that feature values are within expected ranges")
            print("3. Ensure the model is properly loaded")
            print("4. Try reprocessing the input data\n")

def main():
    """
    Main function to demonstrate various error handling scenarios.
    """
    error_demo = ErrorDemo()

    print("=== House Price Prediction System Error Handling Demo ===\n")

    # 1. Demonstrate file not found error
    print("1. Testing File Not Found Error:")
    error_demo.demonstrate_file_not_found('nonexistent_model.joblib')

    # 2. Demonstrate missing dependencies
    print("2. Testing Missing Dependencies:")
    error_demo.demonstrate_missing_dependencies()

    # 3. Demonstrate invalid input
    print("3. Testing Invalid Input:")
    invalid_features = {
        'LotArea': 'invalid',
        'YearBuilt': 2000,
        'BedroomAbvGr': 3,
        # Missing some required features
    }
    error_demo.demonstrate_invalid_input(invalid_features)

    # 4. Demonstrate model loading error
    print("4. Testing Model Loading Error:")
    error_demo.demonstrate_model_load_error('corrupted_model.joblib')

    # 5. Demonstrate prediction error
    print("5. Testing Prediction Error:")
    invalid_prediction_input = {
        'UnknownFeature': 100,
        'LotArea': 8450
    }
    error_demo.demonstrate_prediction_error(invalid_prediction_input)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nUnexpected error occurred!")
        print("Error details:")
        print(traceback.format_exc())
        print("\nSolution:")
        print("1. Check the stack trace above for the source of the error")
        print("2. Verify that all dependencies are correctly installed")
        print("3. Make sure input data is correctly formatted")
        print("4. Contact support if the problem persists") 