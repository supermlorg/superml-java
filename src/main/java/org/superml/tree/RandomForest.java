package org.superml.tree;

import org.superml.core.BaseEstimator;
import org.superml.core.Classifier;
import org.superml.core.Regressor;

import java.util.*;
import java.util.concurrent.*;

/**
 * Random Forest implementation for both classification and regression.
 * Similar to sklearn.ensemble.RandomForestClassifier and RandomForestRegressor
 */
public class RandomForest extends BaseEstimator implements Classifier, Regressor {
    
    // Hyperparameters
    private int nEstimators = 100;
    private String criterion = "gini"; // "gini", "entropy" for classification; "mse" for regression
    private int maxDepth = Integer.MAX_VALUE;
    private int minSamplesSplit = 2;
    private int minSamplesLeaf = 1;
    private double minImpurityDecrease = 0.0;
    private int maxFeatures = -1; // -1 means sqrt(n_features) for classification, n_features for regression
    private boolean bootstrap = true;
    private double maxSamples = 1.0; // Fraction of samples to draw for each tree
    private int randomState = 42;
    private int nJobs = 1; // Number of parallel jobs (-1 for all cores)
    
    // Forest components
    private List<DecisionTree> trees;
    private double[] classes;
    private boolean isClassification = true;
    private boolean fitted = false;
    
    public RandomForest() {
        this.trees = new ArrayList<>();
        
        params.put("n_estimators", nEstimators);
        params.put("criterion", criterion);
        params.put("max_depth", maxDepth);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        params.put("min_impurity_decrease", minImpurityDecrease);
        params.put("max_features", maxFeatures);
        params.put("bootstrap", bootstrap);
        params.put("max_samples", maxSamples);
        params.put("random_state", randomState);
        params.put("n_jobs", nJobs);
    }
    
    public RandomForest(int nEstimators, int maxDepth) {
        this();
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        params.put("n_estimators", nEstimators);
        params.put("max_depth", maxDepth);
    }
    
    @Override
    public RandomForest fit(double[][] X, double[] y) {
        // Determine if this is classification or regression
        isClassification = isClassificationProblem(y);
        
        if (isClassification) {
            Set<Double> uniqueTargets = new HashSet<>();
            for (double target : y) {
                uniqueTargets.add(target);
            }
            classes = uniqueTargets.stream().mapToDouble(Double::doubleValue).sorted().toArray();
            
            if (criterion.equals("mse")) {
                criterion = "gini"; // Default to gini for classification
            }
        } else {
            if (criterion.equals("gini") || criterion.equals("entropy")) {
                criterion = "mse"; // Default to mse for regression
            }
        }
        
        // Set max_features if not specified
        if (maxFeatures == -1) {
            maxFeatures = isClassification ? 
                Math.max(1, (int) Math.sqrt(X[0].length)) : 
                X[0].length;
        }
        
        // Clear previous trees
        trees.clear();
        
        // Determine number of threads
        int numThreads = nJobs == -1 ? Runtime.getRuntime().availableProcessors() : Math.min(nJobs, nEstimators);
        
        if (numThreads > 1 && nEstimators > 1) {
            // Parallel training
            fitParallel(X, y, numThreads);
        } else {
            // Sequential training
            fitSequential(X, y);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Fit trees sequentially.
     */
    private void fitSequential(double[][] X, double[] y) {
        for (int i = 0; i < nEstimators; i++) {
            DecisionTree tree = createTree(i);
            
            if (bootstrap) {
                // Bootstrap sampling
                BootstrapSample sample = createBootstrapSample(X, y, i);
                tree.fit(sample.X, sample.y);
            } else {
                // Use full dataset
                tree.fit(X, y);
            }
            
            trees.add(tree);
        }
    }
    
    /**
     * Fit trees in parallel.
     */
    private void fitParallel(double[][] X, double[] y, int numThreads) {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<DecisionTree>> futures = new ArrayList<>();
        
        for (int i = 0; i < nEstimators; i++) {
            final int treeIndex = i;
            futures.add(executor.submit(() -> {
                DecisionTree tree = createTree(treeIndex);
                
                if (bootstrap) {
                    BootstrapSample sample = createBootstrapSample(X, y, treeIndex);
                    tree.fit(sample.X, sample.y);
                } else {
                    tree.fit(X, y);
                }
                
                return tree;
            }));
        }
        
        // Collect results
        for (Future<DecisionTree> future : futures) {
            try {
                trees.add(future.get());
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Error training tree in parallel", e);
            }
        }
        
        executor.shutdown();
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        if (isClassification) {
            return predictClassification(X);
        } else {
            return predictRegression(X);
        }
    }
    
    /**
     * Predict using majority voting for classification.
     */
    private double[] predictClassification(double[][] X) {
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            Map<Double, Integer> votes = new HashMap<>();
            
            // Collect votes from all trees
            for (DecisionTree tree : trees) {
                double prediction = tree.predict(new double[][]{X[i]})[0];
                votes.merge(prediction, 1, Integer::sum);
            }
            
            // Find class with most votes
            double bestClass = classes[0];
            int maxVotes = 0;
            for (Map.Entry<Double, Integer> entry : votes.entrySet()) {
                if (entry.getValue() > maxVotes) {
                    maxVotes = entry.getValue();
                    bestClass = entry.getKey();
                }
            }
            
            predictions[i] = bestClass;
        }
        
        return predictions;
    }
    
    /**
     * Predict using average for regression.
     */
    private double[] predictRegression(double[][] X) {
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double sum = 0.0;
            
            // Average predictions from all trees
            for (DecisionTree tree : trees) {
                sum += tree.predict(new double[][]{X[i]})[0];
            }
            
            predictions[i] = sum / trees.size();
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        if (!isClassification) {
            throw new IllegalStateException("predict_proba is only available for classification");
        }
        
        double[][] probabilities = new double[X.length][classes.length];
        
        for (int i = 0; i < X.length; i++) {
            double[] avgProba = new double[classes.length];
            
            // Average probabilities from all trees
            for (DecisionTree tree : trees) {
                double[][] treeProba = tree.predictProba(new double[][]{X[i]});
                for (int j = 0; j < classes.length; j++) {
                    avgProba[j] += treeProba[0][j];
                }
            }
            
            // Normalize by number of trees
            for (int j = 0; j < classes.length; j++) {
                probabilities[i][j] = avgProba[j] / trees.size();
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(Math.max(probabilities[i][j], 1e-15));
            }
        }
        return logProbabilities;
    }
    
    @Override
    public double[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classes");
        }
        if (!isClassification) {
            throw new IllegalStateException("getClasses is only available for classification");
        }
        return Arrays.copyOf(classes, classes.length);
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        if (isClassification) {
            // Accuracy for classification
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) {
                    correct++;
                }
            }
            return (double) correct / y.length;
        } else {
            // R² score for regression
            return calculateR2Score(y, predictions);
        }
    }
    
    /**
     * Get feature importances based on impurity decrease.
     */
    public double[] getFeatureImportances() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing feature importances");
        }
        
        // This is a simplified implementation
        // In a full implementation, you'd calculate based on impurity decrease
        int nFeatures = trees.get(0).getParams().containsKey("n_features") ? 
            (Integer) trees.get(0).getParams().get("n_features") : maxFeatures;
        
        double[] importances = new double[nFeatures];
        
        // For now, return uniform importances
        // TODO: Implement proper feature importance calculation
        Arrays.fill(importances, 1.0 / nFeatures);
        
        return importances;
    }
    
    // Helper methods
    
    private boolean isClassificationProblem(double[] y) {
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        return uniqueValues.size() <= Math.sqrt(y.length);
    }
    
    private DecisionTree createTree(int treeIndex) {
        // Create tree with random state based on main random state and tree index
        Random treeRandom = new Random(randomState + treeIndex);
        
        return new DecisionTree(criterion, maxDepth)
                .setMinSamplesSplit(minSamplesSplit)
                .setMinSamplesLeaf(minSamplesLeaf)
                .setMinImpurityDecrease(minImpurityDecrease)
                .setMaxFeatures(maxFeatures)
                .setRandomState(treeRandom.nextInt());
    }
    
    private BootstrapSample createBootstrapSample(double[][] X, double[] y, int seed) {
        Random sampleRandom = new Random(randomState + seed);
        int sampleSize = (int) (X.length * maxSamples);
        
        double[][] sampleX = new double[sampleSize][];
        double[] sampleY = new double[sampleSize];
        
        for (int i = 0; i < sampleSize; i++) {
            int index = sampleRandom.nextInt(X.length);
            sampleX[i] = Arrays.copyOf(X[index], X[index].length);
            sampleY[i] = y[index];
        }
        
        return new BootstrapSample(sampleX, sampleY);
    }
    
    private double calculateR2Score(double[] yTrue, double[] yPred) {
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    // Getters and setters
    
    public int getNEstimators() { return nEstimators; }
    public RandomForest setNEstimators(int nEstimators) {
        this.nEstimators = nEstimators;
        params.put("n_estimators", nEstimators);
        return this;
    }
    
    public String getCriterion() { return criterion; }
    public RandomForest setCriterion(String criterion) {
        this.criterion = criterion;
        params.put("criterion", criterion);
        return this;
    }
    
    public int getMaxDepth() { return maxDepth; }
    public RandomForest setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth;
        params.put("max_depth", maxDepth);
        return this;
    }
    
    public int getMinSamplesSplit() { return minSamplesSplit; }
    public RandomForest setMinSamplesSplit(int minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
        params.put("min_samples_split", minSamplesSplit);
        return this;
    }
    
    public int getMinSamplesLeaf() { return minSamplesLeaf; }
    public RandomForest setMinSamplesLeaf(int minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
        params.put("min_samples_leaf", minSamplesLeaf);
        return this;
    }
    
    public double getMinImpurityDecrease() { return minImpurityDecrease; }
    public RandomForest setMinImpurityDecrease(double minImpurityDecrease) {
        this.minImpurityDecrease = minImpurityDecrease;
        params.put("min_impurity_decrease", minImpurityDecrease);
        return this;
    }
    
    public int getMaxFeatures() { return maxFeatures; }
    public RandomForest setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
        params.put("max_features", maxFeatures);
        return this;
    }
    
    public boolean isBootstrap() { return bootstrap; }
    public RandomForest setBootstrap(boolean bootstrap) {
        this.bootstrap = bootstrap;
        params.put("bootstrap", bootstrap);
        return this;
    }
    
    public double getMaxSamples() { return maxSamples; }
    public RandomForest setMaxSamples(double maxSamples) {
        this.maxSamples = maxSamples;
        params.put("max_samples", maxSamples);
        return this;
    }
    
    public int getRandomState() { return randomState; }
    public RandomForest setRandomState(int randomState) {
        this.randomState = randomState;
        params.put("random_state", randomState);
        return this;
    }
    
    public int getNJobs() { return nJobs; }
    public RandomForest setNJobs(int nJobs) {
        this.nJobs = nJobs;
        params.put("n_jobs", nJobs);
        return this;
    }
    
    /**
     * Get the individual trees in the forest.
     */
    public List<DecisionTree> getTrees() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing trees");
        }
        return new ArrayList<>(trees);
    }
    
    /**
     * Helper class for bootstrap samples.
     */
    private static class BootstrapSample {
        final double[][] X;
        final double[] y;
        
        BootstrapSample(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
        }
    }
}
