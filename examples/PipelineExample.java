package examples;

import com.superml.linear_model.LogisticRegression;
import com.superml.preprocessing.StandardScaler;
import com.superml.pipeline.Pipeline;
import com.superml.model_selection.ModelSelection;
import com.superml.datasets.Datasets;
import com.superml.metrics.Metrics;

/**
 * Pipeline example demonstrating data preprocessing and model training.
 * Shows how to chain preprocessing steps with machine learning models.
 */
public class PipelineExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("       SuperML Java - ML Pipeline Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Load dataset
            System.out.println("Loading Iris dataset...");
            var dataset = Datasets.loadIris();
            
            System.out.printf("Dataset loaded: %d samples, %d features, %d classes\n", 
                            dataset.data.length, dataset.data[0].length, dataset.targetNames.length);
            
            // 2. Split data
            System.out.println("Splitting data (70% train, 30% test)...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.3, 42);
            
            // 3. Create pipeline with preprocessing and model
            System.out.println("Creating ML pipeline...");
            var pipeline = new Pipeline()
                .addStep("scaler", new StandardScaler())
                .addStep("classifier", new LogisticRegression().setMaxIter(1000));
            
            System.out.println("Pipeline steps:");
            for (String stepName : pipeline.getStepNames()) {
                System.out.printf("  • %s\n", stepName);
            }
            
            // 4. Train pipeline
            System.out.println("\nTraining pipeline...");
            long startTime = System.currentTimeMillis();
            pipeline.fit(split.XTrain, split.yTrain);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("✓ Pipeline trained in %d ms\n", trainingTime);
            
            // 5. Make predictions
            System.out.println("Making predictions...");
            startTime = System.currentTimeMillis();
            double[] predictions = pipeline.predict(split.XTest);
            long predictionTime = System.currentTimeMillis() - startTime;
            
            // 6. Evaluate performance
            double accuracy = Metrics.accuracy(split.yTest, predictions);
            double precision = Metrics.precision(split.yTest, predictions, "macro");
            double recall = Metrics.recall(split.yTest, predictions, "macro");
            double f1 = Metrics.f1Score(split.yTest, predictions, "macro");
            
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Pipeline Performance:");
            System.out.println("=".repeat(40));
            System.out.printf("Training time:   %d ms\n", trainingTime);
            System.out.printf("Prediction time: %d ms\n", predictionTime);
            System.out.printf("Accuracy:        %.4f (%.1f%%)\n", accuracy, accuracy * 100);
            System.out.printf("Precision:       %.4f\n", precision);
            System.out.printf("Recall:          %.4f\n", recall);
            System.out.printf("F1-Score:        %.4f\n", f1);
            
            // 7. Show confusion matrix
            System.out.println("\nConfusion Matrix:");
            var confMatrix = Metrics.confusionMatrix(split.yTest, predictions, 3);
            System.out.println("       Predicted");
            System.out.println("      0    1    2");
            for (int i = 0; i < confMatrix.length; i++) {
                System.out.printf("  %d | ", i);
                for (int j = 0; j < confMatrix[i].length; j++) {
                    System.out.printf("%2d   ", confMatrix[i][j]);
                }
                System.out.println();
            }
            
            // 8. Pipeline feature inspection
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Pipeline Feature Analysis:");
            System.out.println("=".repeat(40));
            
            // Get scaler statistics
            var scaler = (StandardScaler) pipeline.getStep("scaler");
            double[] means = scaler.getMean();
            double[] stds = scaler.getScale();
            
            System.out.println("Feature scaling statistics:");
            String[] featureNames = {"Sepal Length", "Sepal Width", "Petal Length", "Petal Width"};
            for (int i = 0; i < means.length; i++) {
                System.out.printf("  %s: mean=%.2f, std=%.2f\n", 
                                featureNames[i], means[i], stds[i]);
            }
            
            // Get model coefficients
            var classifier = (LogisticRegression) pipeline.getStep("classifier");
            if (classifier.getCoefficients() != null) {
                System.out.println("\nModel feature importance (absolute coefficients):");
                double[] coefs = classifier.getCoefficients();
                for (int i = 0; i < Math.min(coefs.length, featureNames.length); i++) {
                    System.out.printf("  %s: %.3f\n", featureNames[i], Math.abs(coefs[i]));
                }
            }
            
            // 9. Sample predictions
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Sample Predictions:");
            System.out.println("=".repeat(40));
            System.out.println("Actual | Predicted | Class Name");
            System.out.println("-".repeat(35));
            
            String[] classNames = dataset.targetNames;
            for (int i = 0; i < Math.min(15, predictions.length); i++) {
                int actual = (int) split.yTest[i];
                int predicted = (int) predictions[i];
                String status = actual == predicted ? "✓" : "✗";
                System.out.printf("  %d    |     %d     | %s %s\n", 
                                actual, predicted, classNames[predicted], status);
            }
            
            System.out.println("\n✓ Pipeline example completed successfully!");
            System.out.println("💡 Pipelines provide a clean way to chain preprocessing and models");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
