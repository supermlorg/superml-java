package org.superml;

import org.superml.cluster.KMeans;
import org.superml.datasets.DataLoaders;
import org.superml.datasets.Datasets;
import org.superml.linear_model.Lasso;
import org.superml.linear_model.LinearRegression;
import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.Ridge;
import org.superml.metrics.Metrics;
import org.superml.model_selection.GridSearchCV;
import org.superml.model_selection.ModelSelection;
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;

import java.util.HashMap;
import java.util.Map;

/**
 * Comprehensive demonstration of SuperML Java Framework capabilities.
 * Shows all major features including Pipeline, Grid Search, and advanced algorithms.
 */
public class SuperMLDemo {
    
    public static void main(String[] args) {
        System.out.println("=================================================");
        System.out.println("    SuperML Java Framework - Comprehensive Demo");
        System.out.println("=================================================\n");
        
        // 1. Basic Classification with Pipeline
        demonstrateClassificationPipeline();
        
        // 2. Regression with Regularization
        demonstrateRegularizedRegression();
        
        // 3. Clustering Analysis
        demonstrateClustering();
        
        // 4. Hyperparameter Tuning with Grid Search
        demonstrateGridSearch();
        
        // 5. Data Loading Capabilities
        demonstrateDataLoading();
        
        // 6. Kaggle Integration Demo (commented out - requires API credentials)
        // demonstrateKaggleIntegration();
        
        System.out.println("\n=================================================");
        System.out.println("    SuperML Java Framework Demo Complete!");
        System.out.println("=================================================");
    }
    
    private static void demonstrateClassificationPipeline() {
        System.out.println("1. CLASSIFICATION WITH PIPELINE");
        System.out.println("================================\n");
        
        // Generate classification dataset
        Datasets.Dataset dataset = Datasets.makeClassification(1000, 20, 2, 42);
        
        // Create pipeline: StandardScaler -> LogisticRegression
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression(0.01, 1000));
        
        System.out.println("Pipeline: " + pipeline);
        
        // Train-test split
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        // Fit the pipeline
        pipeline.fit(split.XTrain, split.yTrain);
        
        // Make predictions
        double[] predictions = pipeline.predict(split.XTest);
        double accuracy = pipeline.score(split.XTest, split.yTest);
        
        System.out.printf("Pipeline Accuracy: %.4f\n", accuracy);
        System.out.printf("Confusion Matrix:\n");
        int[][] confMatrix = Metrics.confusionMatrix(split.yTest, predictions);
        printMatrix(confMatrix);
        
        System.out.printf("Classification Report:\n");
        System.out.printf("Precision: %.4f\n", Metrics.precision(split.yTest, predictions));
        System.out.printf("Recall: %.4f\n", Metrics.recall(split.yTest, predictions));
        System.out.printf("F1-Score: %.4f\n\n", Metrics.f1Score(split.yTest, predictions));
    }
    
    private static void demonstrateRegularizedRegression() {
        System.out.println("2. REGULARIZED REGRESSION COMPARISON");
        System.out.println("====================================\n");
        
        // Generate regression dataset with noise
        Datasets.Dataset dataset = Datasets.makeRegression(500, 10, 0.1, 42);
        
        // Add some noise to make regularization beneficial
        for (int i = 0; i < dataset.target.length; i++) {
            dataset.target[i] += Math.random() * 0.1 - 0.05;
        }
        
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        // Compare different regression models
        LinearRegression linear = new LinearRegression();
        Ridge ridge = new Ridge(1.0);
        Lasso lasso = new Lasso(0.1);
        
        // Train all models
        linear.fit(split.XTrain, split.yTrain);
        ridge.fit(split.XTrain, split.yTrain);
        lasso.fit(split.XTrain, split.yTrain);
        
        // Evaluate
        double linearR2 = linear.score(split.XTest, split.yTest);
        double ridgeR2 = ridge.score(split.XTest, split.yTest);
        double lassoR2 = lasso.score(split.XTest, split.yTest);
        
        System.out.printf("Linear Regression R²: %.4f\n", linearR2);
        System.out.printf("Ridge Regression R²:  %.4f\n", ridgeR2);
        System.out.printf("Lasso Regression R²:  %.4f\n", lassoR2);
        
        // Show coefficient sparsity for Lasso
        double[] lassoCoefs = lasso.getCoefficients();
        int nonZeroCoefs = 0;
        for (double coef : lassoCoefs) {
            if (Math.abs(coef) > 1e-6) nonZeroCoefs++;
        }
        System.out.printf("Lasso selected %d/%d features\n\n", nonZeroCoefs, lassoCoefs.length);
    }
    
    private static void demonstrateClustering() {
        System.out.println("3. CLUSTERING ANALYSIS");
        System.out.println("======================\n");
        
        // Generate clustering dataset
        Datasets.Dataset dataset = Datasets.makeBlobs(300, 4, 2, 1.0, 42);
        
        // Apply K-means clustering
        KMeans kmeans = new KMeans(4, 100, 42);
        kmeans.fit(dataset.data);
        
        // Get cluster assignments
        int[] labels = kmeans.predict(dataset.data);
        
        System.out.println("K-Means Clustering Results:");
        System.out.printf("Number of clusters: %d\n", 4);
        System.out.printf("Inertia (within-cluster sum of squares): %.4f\n", kmeans.getInertia());
        System.out.printf("Dataset size: %d samples\n", dataset.data.length);
        
        // Analyze cluster distribution
        Map<Integer, Integer> clusterCounts = new HashMap<>();
        for (int label : labels) {
            clusterCounts.put(label, clusterCounts.getOrDefault(label, 0) + 1);
        }
        
        System.out.println("Cluster distribution:");
        for (Map.Entry<Integer, Integer> entry : clusterCounts.entrySet()) {
            System.out.printf("  Cluster %d: %d points\n", entry.getKey(), entry.getValue());
        }
        System.out.println();
    }
    
    private static void demonstrateGridSearch() {
        System.out.println("4. HYPERPARAMETER TUNING WITH GRID SEARCH");
        System.out.println("==========================================\n");
        
        // Generate dataset
        Datasets.Dataset dataset = Datasets.makeClassification(800, 15, 2, 42);
        
        // Create parameter grid for LogisticRegression
        Map<String, Object[]> paramGrid = new HashMap<>();
        paramGrid.put("learningRate", new Object[]{0.001, 0.01, 0.1});
        paramGrid.put("maxIterations", new Object[]{500, 1000, 1500});
        
        // Create grid search
        LogisticRegression baseEstimator = new LogisticRegression();
        GridSearchCV gridSearch = new GridSearchCV(baseEstimator, paramGrid, 3, "accuracy", true, 42);
        
        System.out.println("Starting Grid Search...");
        System.out.println(gridSearch.getParameterGridSummary());
        
        // Fit grid search
        gridSearch.fit(dataset.data, dataset.target);
        
        // Results
        System.out.println("\n" + gridSearch.getResultsSummary());
        
        // Use best estimator for final evaluation
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        gridSearch.fit(split.XTrain, split.yTrain);
        double bestScore = gridSearch.score(split.XTest, split.yTest);
        
        System.out.printf("Best estimator test score: %.4f\n\n", bestScore);
    }
    
    private static void demonstrateDataLoading() {
        System.out.println("5. DATA LOADING AND MANIPULATION");
        System.out.println("=================================\n");
        
        // Create a sample dataset and save it
        Datasets.Dataset syntheticData = Datasets.makeRegression(200, 5, 0.1, 42);
        
        // Create feature names
        String[] featureNames = {"feature_1", "feature_2", "feature_3", "feature_4", "feature_5"};
        DataLoaders.Dataset dataset = new DataLoaders.Dataset(
            syntheticData.data, syntheticData.target, featureNames, "target");
        
        // Save to CSV
        String csvPath = "sample_dataset.csv";
        try {
            DataLoaders.saveCsv(dataset, csvPath);
            System.out.println("Saved synthetic dataset to: " + csvPath);
            
            // Load back from CSV
            DataLoaders.Dataset loadedDataset = DataLoaders.loadCsv(csvPath);
            System.out.println("Successfully loaded dataset from CSV");
            
            // Display dataset information
            System.out.println(DataLoaders.getDatasetInfo(loadedDataset));
            
            // Create train-test split files
            DataLoaders.saveTrainTestSplit(loadedDataset, "train_data.csv", "test_data.csv", 0.2, 42);
            System.out.println("Created train/test split files");
            
        } catch (Exception e) {
            System.out.println("Note: CSV file operations may not work in this demo environment");
            System.out.println("In a real application, you can load data like this:");
            System.out.println("  DataLoaders.Dataset data = DataLoaders.loadCsv(\"your_data.csv\");");
            System.out.println("  System.out.println(DataLoaders.getDatasetInfo(data));");
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrate Kaggle integration capabilities.
     * Note: Requires valid Kaggle API credentials.
     */
    @SuppressWarnings("unused")
    private static void demonstrateKaggleIntegration() {
        System.out.println("6. KAGGLE INTEGRATION DEMO");
        System.out.println("==========================\n");
        
        System.out.println("Kaggle Integration Features:");
        System.out.println("✓ Download datasets directly from Kaggle");
        System.out.println("✓ Automatic data loading and preprocessing");
        System.out.println("✓ Smart algorithm selection based on data type");
        System.out.println("✓ Hyperparameter optimization with GridSearch");
        System.out.println("✓ Comprehensive model evaluation");
        
        System.out.println("\nTo use Kaggle integration:");
        System.out.println("1. Get Kaggle API credentials from kaggle.com/account");
        System.out.println("2. Save them to ~/.kaggle/kaggle.json");
        System.out.println("3. Use KaggleTrainingManager for automated ML:");
        
        System.out.println("\nExample usage:");
        System.out.println("// Load credentials");
        System.out.println("KaggleCredentials creds = KaggleCredentials.fromDefaultLocation();");
        System.out.println("");
        System.out.println("// Create training manager");
        System.out.println("KaggleTrainingManager trainer = new KaggleTrainingManager(creds);");
        System.out.println("");
        System.out.println("// Search for datasets");
        System.out.println("trainer.searchDatasets(\"iris\", 5);");
        System.out.println("");
        System.out.println("// Train models automatically");
        System.out.println("List<TrainingResult> results = trainer.trainOnDataset(");
        System.out.println("    \"uciml\", \"iris\", \"species\");");
        System.out.println("");
        System.out.println("// Get best model");
        System.out.println("SupervisedLearner bestModel = trainer.getBestModel(results);");
        
        System.out.println("\nSupported features:");
        System.out.println("• Automatic dataset downloading and extraction");
        System.out.println("• Smart classification vs regression detection");
        System.out.println("• Multiple algorithm comparison (Logistic, Linear, Ridge, Lasso)");
        System.out.println("• Automated hyperparameter tuning");
        System.out.println("• Pipeline creation with preprocessing");
        System.out.println("• Comprehensive evaluation metrics");
        
        System.out.println("\n");
    }
    
    // Utility methods
    
    private static void printMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            System.out.print("  [");
            for (int i = 0; i < row.length; i++) {
                System.out.printf("%3d", row[i]);
                if (i < row.length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
        System.out.println();
    }
}
