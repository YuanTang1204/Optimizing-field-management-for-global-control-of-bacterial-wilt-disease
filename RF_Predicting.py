import pandas as pd
import joblib
import os

def predict_and_save_results():
    # File path configuration
    model_path = r"D:\data\Predict\present\random_forest_model.pkl"
    data_path = r"D:\data\Predict\present\tomato_standardized_data.csv"
    output_path = r"D:\data\Predict\present\prediction_tomato.csv"
    
    try:
        # 1. Load trained model
        print("Loading machine learning model...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # 2. Load prediction dataset
        print("Loading prediction data...")
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully! Data shape: {data.shape}")
        
        # 3. Perform prediction
        print("Performing prediction...")
        predictions = model.predict(data)
        print("Prediction completed!")
        
        # 4. Create dataframe with only prediction results
        results_df = pd.DataFrame(predictions, columns=['Prediction'])
        
        # 5. Save prediction results to CSV file
        print("Saving prediction results...")
        results_df.to_csv(output_path, index=False)
        print(f"Prediction results saved to: {output_path}")
        
        # 6. Display basic information of prediction results
        print(f"\nPrediction result statistics:")
        print(f"Total samples predicted: {len(predictions)}")
        print(f"Prediction results preview:")
        print(results_df.head())
        
        # For classification problems, show counts per class
        if hasattr(model, 'classes_'):
            print(f"\nPrediction counts per class:")
            for i, class_name in enumerate(model.classes_):
                count = (predictions == class_name).sum()
                print(f"Class {class_name}: {count} samples")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

def main():
    print("Starting prediction process...")
    print("=" * 50)
    
    predict_and_save_results()
    
    print("=" * 50)
    print("Prediction process completed!")

if __name__ == "__main__":
    main()