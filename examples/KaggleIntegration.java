package examples;

import org.superml.linear_model.LogisticRegression;
import org.superml.datasets.Datasets;
import org.superml.model_selection.ModelSelection;
import org.superml.metrics.Metrics;
import org.superml.persistence.ModelPersistence;

/**
 * Enhanced Kaggle integration example with proper error handling.
 * Demonstrates complete ML workflow for competition submissions.
 */
public class KaggleIntegration {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("     SuperML Java - Kaggle Competition Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Simulate Kaggle dataset loading
            System.out.println("Loading competition dataset (simulated with Iris)...");
            var dataset = Datasets.loadIris();
            
            System.out.printf("Competition data: %d samples, %d features\n", 
                            dataset.data.length, dataset.data[0].length);
            
            // 2. Data exploration
            System.out.println("\nData Exploration:");
            System.out.println("-".repeat(20));
            exploreData(dataset);
            
            // 3. Data preparation for competition
            System.out.println("\nPreparing data for competition...");
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
            
            System.out.printf("Training set: %d samples\n", split.XTrain.length);
            System.out.printf("Validation set: %d samples\n", split.XTest.length);
            
            // 4. Model training with cross-validation
            System.out.println("\nTraining competition model...");
            var model = new LogisticRegression()
                .setMaxIter(1000)
                .setRegularization("l2")
                .setRegularizationStrength(0.01);
            
            long startTime = System.currentTimeMillis();
            model.fit(split.XTrain, split.yTrain);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("✓ Model trained in %d ms\n", trainingTime);
            
            // 5. Validation and model evaluation
            System.out.println("\nModel Validation:");
            System.out.println("=".repeat(20));
            
            double[] predictions = model.predict(split.XTest);
            
            // Calculate competition metrics
            double accuracy = Metrics.accuracy(split.yTest, predictions);
            double precision = Metrics.precision(split.yTest, predictions, "macro");
            double recall = Metrics.recall(split.yTest, predictions, "macro");
            double f1 = Metrics.f1Score(split.yTest, predictions, "macro");
            
            System.out.printf("Validation Accuracy: %.4f (%.2f%%)\n", accuracy, accuracy * 100);
            System.out.printf("Macro Precision:     %.4f\n", precision);
            System.out.printf("Macro Recall:        %.4f\n", recall);
            System.out.printf("Macro F1-Score:      %.4f\n", f1);
            
            // 6. Cross-validation for robust evaluation
            System.out.println("\nCross-Validation (5-fold):");
            System.out.println("-".repeat(30));
            
            double[] cvScores = performCrossValidation(dataset.data, dataset.target, 5);
            double meanCV = 0;
            for (int i = 0; i < cvScores.length; i++) {
                System.out.printf("Fold %d: %.4f\n", i + 1, cvScores[i]);
                meanCV += cvScores[i];
            }
            meanCV /= cvScores.length;
            
            double stdCV = 0;
            for (double score : cvScores) {
                stdCV += Math.pow(score - meanCV, 2);
            }
            stdCV = Math.sqrt(stdCV / cvScores.length);
            
            System.out.printf("CV Mean: %.4f ± %.4f\n", meanCV, stdCV);
            
            // 7. Feature importance analysis
            System.out.println("\nFeature Importance Analysis:");
            System.out.println("-".repeat(35));
            
            double[] coefficients = model.getCoefficients();
            if (coefficients != null) {
                String[] featureNames = {"Sepal Length", "Sepal Width", "Petal Length", "Petal Width"};
                
                // Calculate feature importance as absolute coefficients
                double maxCoef = 0;
                for (double coef : coefficients) {
                    maxCoef = Math.max(maxCoef, Math.abs(coef));
                }
                
                System.out.println("Feature importance (normalized):");
                for (int i = 0; i < Math.min(coefficients.length, featureNames.length); i++) {
                    double importance = Math.abs(coefficients[i]) / maxCoef;
                    String bar = "█".repeat(Math.max(1, (int)(importance * 20)));
                    System.out.printf("%-15s: %.3f %s\n", featureNames[i], importance, bar);
                }
            }
            
            // 8. Model persistence for submission
            System.out.println("\nPreparing for submission...");
            String modelPath = "examples/kaggle_model.superml";
            ModelPersistence.saveModel(model, modelPath);
            System.out.printf("✓ Model saved to: %s\n", modelPath);
            
            // 9. Generate competition predictions (simulated)
            System.out.println("\nGenerating competition predictions...");
            generateCompetitionSubmission(model, split.XTest, dataset.targetNames);
            
            // 10. Competition summary
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Competition Summary:");
            System.out.println("=".repeat(50));
            System.out.printf("Best CV Score:        %.4f ± %.4f\n", meanCV, stdCV);
            System.out.printf("Validation Accuracy:  %.4f\n", accuracy);
            System.out.printf("Training Time:        %d ms\n", trainingTime);
            System.out.printf("Model Saved:          %s\n", modelPath);
            System.out.println("Submission Ready:     ✓");
            
            // Cleanup
            new java.io.File(modelPath).delete();
            
            System.out.println("\n✓ Kaggle competition example completed successfully!");
            System.out.println("💡 Ready for competition submission with robust validation");
            
        } catch (Exception e) {
            System.err.println("❌ Competition example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void exploreData(Datasets.DatasetBundle dataset) {
        // Basic statistics
        double[][] data = dataset.data;
        int numFeatures = data[0].length;
        
        for (int feature = 0; feature < numFeatures; feature++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            double sum = 0;
            
            for (double[] row : data) {
                double value = row[feature];
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
            }
            
            double mean = sum / data.length;
            System.out.printf("Feature %d: min=%.2f, max=%.2f, mean=%.2f\n", 
                            feature, min, max, mean);
        }
        
        // Class distribution
        int[] classCount = new int[dataset.targetNames.length];
        for (double target : dataset.target) {
            classCount[(int)target]++;
        }
        
        System.out.println("Class distribution:");
        for (int i = 0; i < classCount.length; i++) {
            double percentage = (double) classCount[i] / dataset.target.length * 100;
            System.out.printf("  %s: %d samples (%.1f%%)\n", 
                            dataset.targetNames[i], classCount[i], percentage);
        }
    }
    
    private static double[] performCrossValidation(double[][] X, double[] y, int folds) {
        double[] scores = new double[folds];
        int foldSize = X.length / folds;
        
        for (int fold = 0; fold < folds; fold++) {
            // Create train/validation split for this fold
            int validStart = fold * foldSize;
            int validEnd = (fold == folds - 1) ? X.length : validStart + foldSize;
            
            // Build training and validation sets
            double[][] XTrain = new double[X.length - (validEnd - validStart)][];
            double[] yTrain = new double[X.length - (validEnd - validStart)];
            double[][] XValid = new double[validEnd - validStart][];
            double[] yValid = new double[validEnd - validStart];
            
            int trainIdx = 0, validIdx = 0;
            for (int i = 0; i < X.length; i++) {
                if (i >= validStart && i < validEnd) {
                    XValid[validIdx] = X[i];
                    yValid[validIdx] = y[i];
                    validIdx++;
                } else {
                    XTrain[trainIdx] = X[i];
                    yTrain[trainIdx] = y[i];
                    trainIdx++;
                }
            }
            
            // Train and evaluate
            var model = new LogisticRegression().setMaxIter(1000);
            model.fit(XTrain, yTrain);
            double[] predictions = model.predict(XValid);
            scores[fold] = Metrics.accuracy(yValid, predictions);
        }
        
        return scores;
    }
    
    private static void generateCompetitionSubmission(LogisticRegression model, 
                                                     double[][] testData, 
                                                     String[] classNames) {
        System.out.println("Sample submission format:");
        System.out.println("ID,Prediction");
        
        double[] predictions = model.predict(testData);
        for (int i = 0; i < Math.min(10, predictions.length); i++) {
            int predClass = (int) predictions[i];
            System.out.printf("%d,%s\n", i + 1, classNames[predClass]);
        }
        
        if (predictions.length > 10) {
            System.out.printf("... (%d more rows)\n", predictions.length - 10);
        }
        
        System.out.printf("✓ Generated %d predictions for submission\n", predictions.length);
    }
}
