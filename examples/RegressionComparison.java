package examples;

import com.superml.datasets.Datasets;
import com.superml.linear_model.LinearRegression;
import com.superml.linear_model.Ridge;
import com.superml.linear_model.Lasso;
import com.superml.model_selection.ModelSelection;
import com.superml.metrics.Metrics;

/**
 * Regression example comparing different linear models.
 * Demonstrates Linear Regression, Ridge, and Lasso with performance comparison.
 */
public class RegressionComparison {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("      SuperML Java - Regression Comparison Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Generate synthetic regression dataset
            System.out.println("Generating synthetic regression dataset...");
            var dataset = Datasets.makeRegression(500, 10, 0.1, 42);
            
            System.out.printf("Dataset created: %d samples, %d features\n", 
                            dataset.data.length, dataset.data[0].length);
            
            // 2. Split data
            System.out.println("Splitting data (75% train, 25% test)...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.25, 42);
            
            // 3. Create different regression models
            var linearReg = new LinearRegression();
            var ridgeReg = new Ridge().setAlpha(1.0);
            var lassoReg = new Lasso().setAlpha(0.1);
            
            String[] modelNames = {"Linear Regression", "Ridge Regression", "Lasso Regression"};
            var models = new Object[]{linearReg, ridgeReg, lassoReg};
            
            System.out.println("\nTraining and evaluating models...");
            System.out.println("=".repeat(50));
            
            // 4. Train and evaluate each model
            for (int i = 0; i < models.length; i++) {
                System.out.printf("\n%s:\n", modelNames[i]);
                System.out.println("-".repeat(modelNames[i].length() + 1));
                
                long startTime = System.currentTimeMillis();
                
                // Train model
                if (models[i] instanceof LinearRegression) {
                    ((LinearRegression) models[i]).fit(split.XTrain, split.yTrain);
                } else if (models[i] instanceof Ridge) {
                    ((Ridge) models[i]).fit(split.XTrain, split.yTrain);
                } else if (models[i] instanceof Lasso) {
                    ((Lasso) models[i]).fit(split.XTrain, split.yTrain);
                }
                
                long trainingTime = System.currentTimeMillis() - startTime;
                
                // Make predictions
                double[] predictions = null;
                if (models[i] instanceof LinearRegression) {
                    predictions = ((LinearRegression) models[i]).predict(split.XTest);
                } else if (models[i] instanceof Ridge) {
                    predictions = ((Ridge) models[i]).predict(split.XTest);
                } else if (models[i] instanceof Lasso) {
                    predictions = ((Lasso) models[i]).predict(split.XTest);
                }
                
                // Calculate metrics
                double mse = Metrics.meanSquaredError(split.yTest, predictions);
                double mae = Metrics.meanAbsoluteError(split.yTest, predictions);
                double r2 = Metrics.r2Score(split.yTest, predictions);
                double rmse = Math.sqrt(mse);
                
                // Display results
                System.out.printf("Training time: %d ms\n", trainingTime);
                System.out.printf("R² Score:      %.4f\n", r2);
                System.out.printf("RMSE:          %.4f\n", rmse);
                System.out.printf("MAE:           %.4f\n", mae);
                System.out.printf("MSE:           %.4f\n", mse);
                
                // Show feature coefficients for regularized models
                if (models[i] instanceof Ridge) {
                    double[] coefs = ((Ridge) models[i]).getCoefficients();
                    int nonZeroCoefs = 0;
                    for (double coef : coefs) {
                        if (Math.abs(coef) > 1e-6) nonZeroCoefs++;
                    }
                    System.out.printf("Non-zero coefficients: %d/%d\n", nonZeroCoefs, coefs.length);
                } else if (models[i] instanceof Lasso) {
                    double[] coefs = ((Lasso) models[i]).getCoefficients();
                    int nonZeroCoefs = 0;
                    for (double coef : coefs) {
                        if (Math.abs(coef) > 1e-6) nonZeroCoefs++;
                    }
                    System.out.printf("Non-zero coefficients: %d/%d (feature selection)\n", 
                                    nonZeroCoefs, coefs.length);
                }
            }
            
            // 5. Show sample predictions vs actual
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Sample Predictions (Linear Regression):");
            System.out.println("Actual     | Predicted  | Error");
            System.out.println("-".repeat(35));
            
            double[] linearPredictions = linearReg.predict(split.XTest);
            for (int i = 0; i < Math.min(10, linearPredictions.length); i++) {
                double actual = split.yTest[i];
                double predicted = linearPredictions[i];
                double error = Math.abs(actual - predicted);
                System.out.printf("%9.3f | %10.3f | %6.3f\n", actual, predicted, error);
            }
            
            System.out.println("\n✓ Regression comparison completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
