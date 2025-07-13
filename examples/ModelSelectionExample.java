package org.superml;

import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.Ridge;
import org.superml.model_selection.CrossValidation;
import org.superml.model_selection.HyperparameterTuning;

/**
 * Example demonstrating Cross-Validation and Hyperparameter Tuning capabilities.
 */
public class ModelSelectionExample {
    
    public static void main(String[] args) {
        // Create sample classification data
        double[][] X_classification = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0},
            {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0}, {9.0, 10.0}, {10.0, 11.0},
            {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}, {4.5, 5.5}, {5.5, 6.5},
            {6.5, 7.5}, {7.5, 8.5}, {8.5, 9.5}, {9.5, 10.5}, {10.5, 11.5}
        };
        
        double[] y_classification = {
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1
        };
        
        // Create sample regression data
        double[][] X_regression = {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
            {6.0}, {7.0}, {8.0}, {9.0}, {10.0},
            {11.0}, {12.0}, {13.0}, {14.0}, {15.0}
        };
        
        double[] y_regression = {
            2.1, 4.2, 6.1, 8.3, 10.1,
            12.2, 14.1, 16.3, 18.2, 20.1,
            22.0, 24.1, 26.2, 28.0, 30.1
        };
        
        // 1. CROSS-VALIDATION EXAMPLES
        System.out.println("=== CROSS-VALIDATION EXAMPLES ===\n");
        
        // Basic cross-validation with default configuration
        System.out.println("1. Basic Cross-Validation (Classification):");
        LogisticRegression classifier = new LogisticRegression();
        CrossValidation.CrossValidationResults cvResults = 
            CrossValidation.crossValidate(classifier, X_classification, y_classification);
        System.out.println(cvResults);
        System.out.println();
        
        // Cross-validation with custom configuration
        System.out.println("2. Custom Cross-Validation Configuration:");
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setFolds(3)
                .setShuffle(true)
                .setRandomSeed(42L)
                .setMetrics("accuracy", "precision", "recall", "f1");
        
        CrossValidation.CrossValidationResults customCvResults = 
            CrossValidation.crossValidate(classifier, X_classification, y_classification, config);
        System.out.println(customCvResults);
        System.out.println();
        
        // Regression cross-validation
        System.out.println("3. Regression Cross-Validation:");
        Ridge regressor = new Ridge();
        CrossValidation.CrossValidationResults regressionResults = 
            CrossValidation.crossValidateRegression(regressor, X_regression, y_regression, 
                new CrossValidation.CrossValidationConfig());
        System.out.println(regressionResults);
        System.out.println();
        
        // 2. HYPERPARAMETER TUNING EXAMPLES
        System.out.println("=== HYPERPARAMETER TUNING EXAMPLES ===\n");
        
        // Grid Search for classification
        System.out.println("4. Grid Search (Classification):");
        HyperparameterTuning.TuningResults gridResults = HyperparameterTuning.gridSearch(
            new LogisticRegression(),
            X_classification,
            y_classification,
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1, 0.5),
            HyperparameterTuning.ParameterSpec.discrete("maxIterations", 100, 500, 1000)
        );
        System.out.println(gridResults);
        System.out.println();
        
        // Grid Search for regression
        System.out.println("5. Grid Search (Regression):");
        HyperparameterTuning.TuningResults gridRegressionResults = 
            HyperparameterTuning.gridSearchRegressor(
                new Ridge(),
                X_regression,
                y_regression,
                HyperparameterTuning.ParameterSpec.discrete("alpha", 0.1, 1.0, 10.0),
                HyperparameterTuning.ParameterSpec.discrete("maxIter", 500, 1000)
            );
        System.out.println(gridRegressionResults);
        System.out.println();
        
        // Random Search
        System.out.println("6. Random Search:");
        HyperparameterTuning.TuningResults randomResults = HyperparameterTuning.randomSearch(
            new LogisticRegression(),
            X_classification,
            y_classification,
            5, // number of iterations
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.001, 0.01, 0.05, 0.1, 0.2, 0.5),
            HyperparameterTuning.ParameterSpec.discrete("maxIterations", 50, 100, 200, 500, 1000)
        );
        System.out.println(randomResults);
        System.out.println();
        
        // Advanced Grid Search with custom configuration
        System.out.println("7. Advanced Grid Search with Custom Configuration:");
        HyperparameterTuning.TuningConfig advancedConfig = 
            new HyperparameterTuning.TuningConfig()
                .setScoringMetric("f1")
                .setCvFolds(3)
                .setParallel(true)
                .setVerbose(true)
                .setRandomSeed(123L);
        
        HyperparameterTuning.TuningResults advancedResults = 
            HyperparameterTuning.GridSearch.search(
                new LogisticRegression(),
                X_classification,
                y_classification,
                java.util.Arrays.asList(
                    HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1),
                    HyperparameterTuning.ParameterSpec.discrete("maxIterations", 100, 500)
                ),
                advancedConfig
            );
        System.out.println(advancedResults);
        System.out.println();
        
        // Parameter specifications examples
        System.out.println("8. Parameter Specification Examples:");
        
        // Discrete parameters
        HyperparameterTuning.ParameterSpec discrete = 
            HyperparameterTuning.ParameterSpec.discrete("categories", "A", "B", "C");
        System.out.println("Discrete parameter: " + discrete.getName() + 
                          " with " + discrete.getValues().length + " values");
        
        // Continuous parameters
        HyperparameterTuning.ParameterSpec continuous = 
            HyperparameterTuning.ParameterSpec.continuous("learning_rate", 0.001, 0.1, 10);
        System.out.println("Continuous parameter: " + continuous.getName() + 
                          " with " + continuous.getValues().length + " values");
        
        // Integer parameters
        HyperparameterTuning.ParameterSpec integer = 
            HyperparameterTuning.ParameterSpec.integer("max_depth", 1, 10);
        System.out.println("Integer parameter: " + integer.getName() + 
                          " with " + integer.getValues().length + " values");
        System.out.println();
        
        // Performance comparison example
        System.out.println("=== PERFORMANCE COMPARISON ===\n");
        
        System.out.println("9. Model Performance Comparison:");
        
        // Compare different models using cross-validation
        LogisticRegression lr1 = new LogisticRegression().setLearningRate(0.01).setMaxIter(500);
        LogisticRegression lr2 = new LogisticRegression().setLearningRate(0.1).setMaxIter(1000);
        
        CrossValidation.CrossValidationResults results1 = 
            CrossValidation.crossValidate(lr1, X_classification, y_classification);
        CrossValidation.CrossValidationResults results2 = 
            CrossValidation.crossValidate(lr2, X_classification, y_classification);
        
        System.out.printf("Model 1 (lr=0.01, iter=500): Accuracy = %.4f ± %.4f\n", 
                         results1.getMeanScore("accuracy"), results1.getStdScore("accuracy"));
        System.out.printf("Model 2 (lr=0.1, iter=1000): Accuracy = %.4f ± %.4f\n", 
                         results2.getMeanScore("accuracy"), results2.getStdScore("accuracy"));
        
        if (results1.getMeanScore("accuracy") > results2.getMeanScore("accuracy")) {
            System.out.println("Model 1 performs better!");
        } else {
            System.out.println("Model 2 performs better!");
        }
    }
}
