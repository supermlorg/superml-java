package examples;

import org.superml.linear_model.LogisticRegression;
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;
import org.superml.model_selection.ModelSelection;
import org.superml.datasets.Datasets;
import org.superml.metrics.Metrics;

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
            // 1. Generate dataset
            System.out.println("Generating classification dataset...");
            var dataset = Datasets.makeClassification(200, 4, 3, 42);
            
            System.out.printf("Dataset generated: %d samples, %d features, %d classes\n", 
                            dataset.data.length, dataset.data[0].length, 3);
            
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
            double precision = Metrics.precision(split.yTest, predictions);
            double recall = Metrics.recall(split.yTest, predictions);
            double f1 = Metrics.f1Score(split.yTest, predictions);
            
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
            
            // 8. Pipeline feature inspection
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Pipeline Feature Analysis:");
            System.out.println("=".repeat(40));
            
            // Get scaler statistics
            var scaler = (StandardScaler) pipeline.getStep("scaler");
            double[] means = scaler.getMean();
            double[] stds = scaler.getScale();
            
            System.out.println("Feature scaling statistics:");
            String[] featureNames = {"Feature 1", "Feature 2", "Feature 3", "Feature 4"};
            for (int i = 0; i < means.length; i++) {
                System.out.printf("  %s: mean=%.2f, std=%.2f\n", 
                                featureNames[i], means[i], stds[i]);
            }
            
            System.out.println("\nPipeline successfully trained with both preprocessing and classification");
            
            // 9. Sample predictions
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Sample Predictions:");
            System.out.println("=".repeat(40));
            System.out.println("Actual | Predicted | Status");
            System.out.println("-".repeat(30));
            
            String[] classNames = {"Class A", "Class B", "Class C"};
            for (int i = 0; i < Math.min(12, predictions.length); i++) {
                int actual = (int) split.yTest[i];
                int predicted = (int) predictions[i];
                String status = actual == predicted ? "✓" : "✗";
                System.out.printf("  %d    |     %d     | %s (%s)\n", 
                                actual, predicted, status, classNames[predicted]);
            }
            
            System.out.println("\n✓ Pipeline example completed successfully!");
            System.out.println("💡 Pipelines provide a clean way to chain preprocessing and models");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
