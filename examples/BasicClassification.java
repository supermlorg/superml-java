package examples;

import com.superml.datasets.Datasets;
import com.superml.linear_model.LogisticRegression;
import com.superml.model_selection.ModelSelection;
import com.superml.metrics.Metrics;

import java.util.Arrays;

/**
 * Basic classification example using the Iris dataset.
 * Demonstrates fundamental SuperML concepts: loading data, training, and evaluation.
 */
public class BasicClassification {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("        SuperML Java - Basic Classification Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Load the famous Iris dataset
            System.out.println("Loading Iris dataset...");
            var dataset = Datasets.loadIris();
            
            System.out.printf("Dataset loaded: %d samples, %d features\n", 
                            dataset.data.length, dataset.data[0].length);
            System.out.println("Target classes: " + Arrays.toString(dataset.targetNames));
            
            // 2. Split into training and test sets
            System.out.println("\nSplitting data (80% train, 20% test)...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
            
            System.out.printf("Training samples: %d\n", split.XTrain.length);
            System.out.printf("Test samples: %d\n", split.XTest.length);
            
            // 3. Create and train classifier
            System.out.println("\nTraining Logistic Regression classifier...");
            var classifier = new LogisticRegression()
                .setMaxIter(1000)
                .setLearningRate(0.01);
            
            long startTime = System.currentTimeMillis();
            classifier.fit(split.XTrain, split.yTrain);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Training completed in %d ms\n", trainingTime);
            
            // 4. Make predictions
            System.out.println("\nMaking predictions on test set...");
            double[] predictions = classifier.predict(split.XTest);
            
            // 5. Evaluate performance
            double accuracy = Metrics.accuracy(split.yTest, predictions);
            double precision = Metrics.precision(split.yTest, predictions);
            double recall = Metrics.recall(split.yTest, predictions);
            double f1 = Metrics.f1Score(split.yTest, predictions);
            
            System.out.println("\n" + "=".repeat(30));
            System.out.println("        RESULTS");
            System.out.println("=".repeat(30));
            System.out.printf("Accuracy:  %.3f (%.1f%%)\n", accuracy, accuracy * 100);
            System.out.printf("Precision: %.3f\n", precision);
            System.out.printf("Recall:    %.3f\n", recall);
            System.out.printf("F1-Score:  %.3f\n", f1);
            
            // 6. Show confusion matrix
            int[][] confMatrix = Metrics.confusionMatrix(split.yTest, predictions);
            System.out.println("\nConfusion Matrix:");
            for (int i = 0; i < confMatrix.length; i++) {
                System.out.printf("Class %d: %s\n", i, Arrays.toString(confMatrix[i]));
            }
            
            // 7. Show some sample predictions
            System.out.println("\nSample Predictions:");
            System.out.println("Actual | Predicted | Sample Features");
            System.out.println("-".repeat(40));
            for (int i = 0; i < Math.min(10, split.XTest.length); i++) {
                System.out.printf("  %3.0f  |    %3.0f    | [%.1f, %.1f, %.1f, %.1f]\n",
                    split.yTest[i], predictions[i],
                    split.XTest[i][0], split.XTest[i][1], split.XTest[i][2], split.XTest[i][3]);
            }
            
            System.out.println("\n✓ Classification example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
