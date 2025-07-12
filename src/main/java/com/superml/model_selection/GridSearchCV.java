package com.superml.model_selection;

import com.superml.core.BaseEstimator;
import com.superml.core.SupervisedLearner;

import java.util.*;

/**
 * Grid Search Cross Validation for hyperparameter tuning.
 * Similar to sklearn.model_selection.GridSearchCV
 */
public class GridSearchCV extends BaseEstimator implements SupervisedLearner {
    
    private SupervisedLearner estimator;
    private Map<String, Object[]> paramGrid;
    private int cv;
    private String scoring;
    private boolean refit;
    private int randomState;
    
    // Results
    private SupervisedLearner bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore;
    private List<GridSearchResult> cvResults;
    private boolean fitted = false;
    
    /**
     * Container for grid search results.
     */
    public static class GridSearchResult {
        public final Map<String, Object> params;
        public final double meanTestScore;
        public final double stdTestScore;
        public final double[] testScores;
        public final double meanTrainScore;
        public final double stdTrainScore;
        public final double[] trainScores;
        public final int rank;
        
        public GridSearchResult(Map<String, Object> params, double[] testScores, double[] trainScores, int rank) {
            this.params = new HashMap<>(params);
            this.testScores = testScores.clone();
            this.trainScores = trainScores.clone();
            this.rank = rank;
            
            this.meanTestScore = Arrays.stream(testScores).average().orElse(0.0);
            this.stdTestScore = calculateStd(testScores, meanTestScore);
            this.meanTrainScore = Arrays.stream(trainScores).average().orElse(0.0);
            this.stdTrainScore = calculateStd(trainScores, meanTrainScore);
        }
        
        private double calculateStd(double[] values, double mean) {
            double variance = Arrays.stream(values)
                .map(x -> Math.pow(x - mean, 2))
                .average()
                .orElse(0.0);
            return Math.sqrt(variance);
        }
    }
    
    /**
     * Constructor with default parameters.
     * @param estimator estimator to tune
     * @param paramGrid parameter grid
     */
    public GridSearchCV(SupervisedLearner estimator, Map<String, Object[]> paramGrid) {
        this(estimator, paramGrid, 5, "accuracy", true, 42);
    }
    
    /**
     * Full constructor.
     * @param estimator estimator to tune
     * @param paramGrid parameter grid
     * @param cv number of cross-validation folds
     * @param scoring scoring metric
     * @param refit whether to refit on best parameters
     * @param randomState random seed
     */
    public GridSearchCV(SupervisedLearner estimator, Map<String, Object[]> paramGrid, 
                       int cv, String scoring, boolean refit, int randomState) {
        this.estimator = estimator;
        this.paramGrid = new HashMap<>(paramGrid);
        this.cv = cv;
        this.scoring = scoring;
        this.refit = refit;
        this.randomState = randomState;
        this.cvResults = new ArrayList<>();
    }
    
    /**
     * Fit the grid search.
     * @param X training data
     * @param y target values
     * @return this grid search instance
     */
    public GridSearchCV fit(double[][] X, double[] y) {
        List<Map<String, Object>> paramCombinations = generateParameterCombinations();
        cvResults.clear();
        
        bestScore = Double.NEGATIVE_INFINITY;
        bestParams = null;
        bestEstimator = null;
        
        // Evaluate each parameter combination
        for (int i = 0; i < paramCombinations.size(); i++) {
            Map<String, Object> params = paramCombinations.get(i);
            
            // Create a copy of the estimator with these parameters
            SupervisedLearner estimatorCopy = createEstimatorCopy(params);
            
            // Perform cross-validation
            double[] testScores = crossValidate(estimatorCopy, X, y);
            double[] trainScores = crossValidateTraining(estimatorCopy, X, y);
            
            double meanTestScore = Arrays.stream(testScores).average().orElse(0.0);
            
            // Update best parameters if this is the best score so far
            if (meanTestScore > bestScore) {
                bestScore = meanTestScore;
                bestParams = new HashMap<>(params);
                
                if (refit) {
                    bestEstimator = createEstimatorCopy(params);
                    bestEstimator.fit(X, y);
                }
            }
            
            // Store results
            cvResults.add(new GridSearchResult(params, testScores, trainScores, 0));
        }
        
        // Rank results by test score
        cvResults.sort((a, b) -> Double.compare(b.meanTestScore, a.meanTestScore));
        for (int i = 0; i < cvResults.size(); i++) {
            GridSearchResult result = cvResults.get(i);
            cvResults.set(i, new GridSearchResult(result.params, result.testScores, result.trainScores, i + 1));
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Make predictions using the best estimator.
     * @param X test data
     * @return predictions
     */
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("GridSearchCV must be fitted before making predictions");
        }
        if (bestEstimator == null) {
            throw new IllegalStateException("No best estimator found or refit=false");
        }
        
        return bestEstimator.predict(X);
    }
    
    /**
     * Score using the best estimator.
     * @param X test data
     * @param y true values
     * @return score
     */
    public double score(double[][] X, double[] y) {
        if (!fitted) {
            throw new IllegalStateException("GridSearchCV must be fitted before scoring");
        }
        if (bestEstimator == null) {
            throw new IllegalStateException("No best estimator found or refit=false");
        }
        
        return bestEstimator.score(X, y);
    }
    
    /**
     * Generate all parameter combinations from the grid.
     * @return list of parameter combinations
     */
    private List<Map<String, Object>> generateParameterCombinations() {
        List<Map<String, Object>> combinations = new ArrayList<>();
        
        List<String> paramNames = new ArrayList<>(paramGrid.keySet());
        if (paramNames.isEmpty()) {
            combinations.add(new HashMap<>());
            return combinations;
        }
        
        generateCombinationsRecursive(paramNames, 0, new HashMap<>(), combinations);
        return combinations;
    }
    
    /**
     * Recursively generate parameter combinations.
     */
    private void generateCombinationsRecursive(List<String> paramNames, int index, 
                                             Map<String, Object> current, 
                                             List<Map<String, Object>> combinations) {
        if (index == paramNames.size()) {
            combinations.add(new HashMap<>(current));
            return;
        }
        
        String paramName = paramNames.get(index);
        Object[] values = paramGrid.get(paramName);
        
        for (Object value : values) {
            current.put(paramName, value);
            generateCombinationsRecursive(paramNames, index + 1, current, combinations);
        }
        
        current.remove(paramName);
    }
    
    /**
     * Create a copy of the estimator with specific parameters.
     */
    private SupervisedLearner createEstimatorCopy(Map<String, Object> params) {
        try {
            // Create a new instance of the same class
            SupervisedLearner copy = estimator.getClass().getDeclaredConstructor().newInstance();
            
            // Set the parameters
            copy.setParams(params);
            
            return copy;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create estimator copy: " + e.getMessage(), e);
        }
    }
    
    /**
     * Perform cross-validation and return test scores.
     */
    private double[] crossValidate(SupervisedLearner estimator, double[][] X, double[] y) {
        ModelSelection.KFold kfold = new ModelSelection.KFold(cv, true, randomState);
        List<int[][]> indexSplits = kfold.split(X.length);
        
        double[] scores = new double[indexSplits.size()];
        
        for (int i = 0; i < indexSplits.size(); i++) {
            int[] trainIndices = indexSplits.get(i)[0];
            int[] testIndices = indexSplits.get(i)[1];
            
            // Extract training and test data
            double[][] XTrain = extractRows(X, trainIndices);
            double[] yTrain = extractElements(y, trainIndices);
            double[][] XTest = extractRows(X, testIndices);
            double[] yTest = extractElements(y, testIndices);
            
            // Create fresh estimator for each fold
            SupervisedLearner foldEstimator = createEstimatorCopy(estimator.getParams());
            
            // Fit on training data
            foldEstimator.fit(XTrain, yTrain);
            
            // Score on validation data
            scores[i] = foldEstimator.score(XTest, yTest);
        }
        
        return scores;
    }
    
    /**
     * Perform cross-validation and return training scores.
     */
    private double[] crossValidateTraining(SupervisedLearner estimator, double[][] X, double[] y) {
        ModelSelection.KFold kfold = new ModelSelection.KFold(cv, true, randomState);
        List<int[][]> indexSplits = kfold.split(X.length);
        
        double[] scores = new double[indexSplits.size()];
        
        for (int i = 0; i < indexSplits.size(); i++) {
            int[] trainIndices = indexSplits.get(i)[0];
            
            // Extract training data
            double[][] XTrain = extractRows(X, trainIndices);
            double[] yTrain = extractElements(y, trainIndices);
            
            // Create fresh estimator for each fold
            SupervisedLearner foldEstimator = createEstimatorCopy(estimator.getParams());
            
            // Fit and score on training data
            foldEstimator.fit(XTrain, yTrain);
            scores[i] = foldEstimator.score(XTrain, yTrain);
        }
        
        return scores;
    }
    
    /**
     * Extract specific rows from a 2D array.
     */
    private double[][] extractRows(double[][] data, int[] indices) {
        double[][] result = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            result[i] = data[indices[i]].clone();
        }
        return result;
    }
    
    /**
     * Extract specific elements from a 1D array.
     */
    private double[] extractElements(double[] data, int[] indices) {
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = data[indices[i]];
        }
        return result;
    }
    
    // Getters
    
    public SupervisedLearner getBestEstimator() {
        return bestEstimator;
    }
    
    public Map<String, Object> getBestParams() {
        return bestParams != null ? new HashMap<>(bestParams) : null;
    }
    
    public double getBestScore() {
        return bestScore;
    }
    
    public List<GridSearchResult> getCvResults() {
        return new ArrayList<>(cvResults);
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    // Parameter setters
    
    public GridSearchCV setCv(int cv) {
        this.cv = cv;
        return this;
    }
    
    public GridSearchCV setScoring(String scoring) {
        this.scoring = scoring;
        return this;
    }
    
    public GridSearchCV setRefit(boolean refit) {
        this.refit = refit;
        return this;
    }
    
    public GridSearchCV setRandomState(int randomState) {
        this.randomState = randomState;
        return this;
    }
    
    /**
     * Get parameter grid summary.
     */
    public String getParameterGridSummary() {
        StringBuilder sb = new StringBuilder("Parameter Grid:\n");
        
        for (Map.Entry<String, Object[]> entry : paramGrid.entrySet()) {
            sb.append("  ").append(entry.getKey()).append(": ");
            sb.append(Arrays.toString(entry.getValue())).append("\n");
        }
        
        int totalCombinations = 1;
        for (Object[] values : paramGrid.values()) {
            totalCombinations *= values.length;
        }
        
        sb.append("Total combinations: ").append(totalCombinations);
        return sb.toString();
    }
    
    /**
     * Get results summary.
     */
    public String getResultsSummary() {
        if (!fitted) {
            return "GridSearchCV not fitted yet";
        }
        
        StringBuilder sb = new StringBuilder("Grid Search Results:\n");
        sb.append("Best Score: ").append(String.format("%.4f", bestScore)).append("\n");
        sb.append("Best Parameters: ").append(bestParams).append("\n");
        
        sb.append("\nTop 5 Results:\n");
        int limit = Math.min(5, cvResults.size());
        for (int i = 0; i < limit; i++) {
            GridSearchResult result = cvResults.get(i);
            sb.append(String.format("Rank %d: %.4f (±%.4f) - %s\n", 
                result.rank, result.meanTestScore, result.stdTestScore, result.params));
        }
        
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return String.format("GridSearchCV(estimator=%s, cv=%d, scoring=%s)", 
            estimator.getClass().getSimpleName(), cv, scoring);
    }
}
