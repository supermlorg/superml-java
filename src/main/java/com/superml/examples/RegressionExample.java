package com.superml.examples;

import com.superml.datasets.Datasets;
import com.superml.linear_model.LinearRegression;
import com.superml.model_selection.ModelSelection;
import com.superml.metrics.Metrics;

/**
 * Regression example demonstrating linear regression.
 * Shows how to train and evaluate regression models.
 */
public class RegressionExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("      SuperML Java - Regression Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Generate synthetic regression dataset
            System.out.println("Generating synthetic regression dataset...");
            var dataset = Datasets.makeRegression(500, 5, 0.1, 42);
            
            System.out.printf("Dataset created: %d samples, %d features\n", 
                            dataset.data.length, dataset.data[0].length);
            
            // 2. Split data
            System.out.println("Splitting data (75% train, 25% test)...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.25, 42);
            
            // 3. Create and train linear regression model
            System.out.println("\nTraining Linear Regression model...");
            var model = new LinearRegression();
            
            long startTime = System.currentTimeMillis();
            model.fit(split.XTrain, split.yTrain);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("✓ Model trained in %d ms\n", trainingTime);
            
            // 4. Make predictions
            System.out.println("Making predictions on test set...");
            double[] predictions = model.predict(split.XTest);
            
            // 5. Evaluate performance
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Model Performance:");
            System.out.println("=".repeat(40));
            
            double mse = Metrics.meanSquaredError(split.yTest, predictions);
            double mae = Metrics.meanAbsoluteError(split.yTest, predictions);
            double r2 = Metrics.r2Score(split.yTest, predictions);
            double rmse = Math.sqrt(mse);
            
            System.out.printf("Training time:   %d ms\n", trainingTime);
            System.out.printf("R² Score:        %.4f\n", r2);
            System.out.printf("RMSE:            %.4f\n", rmse);
            System.out.printf("MAE:             %.4f\n", mae);
            System.out.printf("MSE:             %.4f\n", mse);
            
            // 6. Show sample predictions vs actual
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Sample Predictions:");
            System.out.println("=".repeat(40));
            System.out.println("Actual     | Predicted  | Error");
            System.out.println("-".repeat(35));
            
            for (int i = 0; i < Math.min(15, predictions.length); i++) {
                double actual = split.yTest[i];
                double predicted = predictions[i];
                double error = Math.abs(actual - predicted);
                System.out.printf("%9.3f | %10.3f | %6.3f\n", actual, predicted, error);
            }
            
            // 7. Model inspection
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Model Analysis:");
            System.out.println("=".repeat(40));
            
            double[] coefficients = model.getCoefficients();
            if (coefficients != null) {
                System.out.println("Feature coefficients:");
                for (int i = 0; i < coefficients.length; i++) {
                    System.out.printf("  Feature %d: %8.4f\n", i + 1, coefficients[i]);
                }
                
                double intercept = model.getIntercept();
                System.out.printf("Intercept:   %8.4f\n", intercept);
            }
            
            System.out.println("\n✓ Regression example completed successfully!");
            System.out.println("💡 Linear regression works well for linearly separable data");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
