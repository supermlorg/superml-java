package com.superml.examples;

import com.superml.datasets.Datasets;
import com.superml.linear_model.LogisticRegression;
import com.superml.model_selection.ModelSelection;
import com.superml.metrics.Metrics;

/**
 * Basic classification example using synthetic data.
 * Demonstrates a complete machine learning workflow from data loading to evaluation.
 */
public class BasicClassification {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("     SuperML Java - Basic Classification Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Generate a synthetic classification dataset (similar to Iris)
            System.out.println("Generating synthetic classification dataset...");
            var dataset = Datasets.makeClassification(150, 4, 3, 42);
            
            // Create feature and class names for demonstration
            String[] featureNames = {"Feature 1", "Feature 2", "Feature 3", "Feature 4"};
            String[] classNames = {"Class A", "Class B", "Class C"};
            
            System.out.printf("Dataset generated: %d samples, %d features, %d classes\n", 
                            dataset.data.length, dataset.data[0].length, classNames.length);
            System.out.printf("Classes: %s\n", String.join(", ", classNames));
            
            // 2. Split the data into training and testing sets
            System.out.println("\nSplitting data (80% train, 20% test)...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
            
            System.out.printf("Training set: %d samples\n", split.XTrain.length);
            System.out.printf("Test set: %d samples\n", split.XTest.length);
            
            // 3. Create and train the model
            System.out.println("\nTraining Logistic Regression model...");
            var model = new LogisticRegression().setMaxIter(1000);
            
            long startTime = System.currentTimeMillis();
            model.fit(split.XTrain, split.yTrain);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("✓ Model trained in %d ms\n", trainingTime);
            
            // 4. Make predictions
            System.out.println("Making predictions on test set...");
            double[] predictions = model.predict(split.XTest);
            
            // 5. Evaluate the model
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Model Performance:");
            System.out.println("=".repeat(40));
            
            double accuracy = Metrics.accuracy(split.yTest, predictions);
            double precision = Metrics.precision(split.yTest, predictions);
            double recall = Metrics.recall(split.yTest, predictions);
            double f1 = Metrics.f1Score(split.yTest, predictions);
            
            System.out.printf("Accuracy:        %.4f (%.1f%%)\n", accuracy, accuracy * 100);
            System.out.printf("Precision:       %.4f\n", precision);
            System.out.printf("Recall:          %.4f\n", recall);
            System.out.printf("F1-Score:        %.4f\n", f1);
            
            // 6. Show confusion matrix
            System.out.println("\nConfusion Matrix:");
            int[][] confMatrix = Metrics.confusionMatrix(split.yTest, predictions);
            System.out.println("       Predicted");
            System.out.print("      ");
            for (int i = 0; i < confMatrix.length; i++) {
                System.out.printf("%d    ", i);
            }
            System.out.println();
            for (int i = 0; i < confMatrix.length; i++) {
                System.out.printf("  %d | ", i);
                for (int j = 0; j < confMatrix[i].length; j++) {
                    System.out.printf("%2d   ", confMatrix[i][j]);
                }
                System.out.println();
            }
            
            // 7. Show sample predictions
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Sample Predictions:");
            System.out.println("=".repeat(40));
            System.out.println("Actual | Predicted | Class Name");
            System.out.println("-".repeat(35));
            
            for (int i = 0; i < Math.min(10, predictions.length); i++) {
                int actual = (int) split.yTest[i];
                int predicted = (int) predictions[i];
                String className = classNames[predicted];
                String status = actual == predicted ? "✓" : "✗";
                System.out.printf("  %d    |     %d     | %s %s\n", actual, predicted, className, status);
            }
            
            System.out.println("\n✓ Classification example completed successfully!");
            System.out.println("💡 Try experimenting with different algorithms and parameters!");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
