package org.superml.tree;

import org.superml.core.BaseEstimator;
import org.superml.core.Classifier;
import org.superml.core.Regressor;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Decision Tree implementation for both classification and regression.
 * Similar to sklearn.tree.DecisionTreeClassifier and DecisionTreeRegressor
 */
public class DecisionTree extends BaseEstimator implements Classifier, Regressor {
    
    // Hyperparameters
    private String criterion = "gini"; // "gini", "entropy" for classification; "mse" for regression
    private int maxDepth = Integer.MAX_VALUE;
    private int minSamplesSplit = 2;
    private int minSamplesLeaf = 1;
    private double minImpurityDecrease = 0.0;
    private int maxFeatures = -1; // -1 means use all features
    private Random random;
    private int randomState = 42;
    private boolean isClassification = true;
    
    // Tree structure
    private TreeNode root;
    private double[] classes;
    private boolean fitted = false;
    
    /**
     * Internal class representing a node in the decision tree.
     */
    public static class TreeNode {
        // Split criteria
        public int featureIndex = -1;
        public double threshold = 0.0;
        public double impurity = 0.0;
        public int samples = 0;
        
        // For leaves
        public double[] value; // For classification: class probabilities, for regression: mean value
        public double prediction; // Single prediction value
        
        // Children
        public TreeNode left;
        public TreeNode right;
        
        public boolean isLeaf() {
            return left == null && right == null;
        }
    }
    
    public DecisionTree() {
        this.random = new Random(randomState);
        params.put("criterion", criterion);
        params.put("max_depth", maxDepth);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        params.put("min_impurity_decrease", minImpurityDecrease);
        params.put("random_state", randomState);
    }
    
    public DecisionTree(String criterion, int maxDepth) {
        this();
        this.criterion = criterion;
        this.maxDepth = maxDepth;
        params.put("criterion", criterion);
        params.put("max_depth", maxDepth);
    }
    
    @Override
    public DecisionTree fit(double[][] X, double[] y) {
        // Determine if this is classification or regression
        Set<Double> uniqueTargets = new HashSet<>();
        for (double target : y) {
            uniqueTargets.add(target);
        }
        
        isClassification = isClassificationProblem(y);
        
        if (isClassification) {
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
            maxFeatures = X[0].length;
        }
        
        // Build the tree
        int[] sampleIndices = IntStream.range(0, X.length).toArray();
        root = buildTree(X, y, sampleIndices, 0);
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSample(X[i]);
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
            double[] proba = predictProbaSample(X[i]);
            System.arraycopy(proba, 0, probabilities[i], 0, proba.length);
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
     * Build the decision tree recursively.
     */
    private TreeNode buildTree(double[][] X, double[] y, int[] sampleIndices, int depth) {
        TreeNode node = new TreeNode();
        node.samples = sampleIndices.length;
        
        // Calculate impurity and value for this node
        if (isClassification) {
            node.value = calculateClassProbabilities(y, sampleIndices);
            node.prediction = classes[getMaxIndex(node.value)];
            node.impurity = calculateImpurity(y, sampleIndices, criterion);
        } else {
            double mean = calculateMean(y, sampleIndices);
            node.value = new double[]{mean};
            node.prediction = mean;
            node.impurity = calculateMse(y, sampleIndices, mean);
        }
        
        // Check stopping criteria
        if (shouldStop(depth, sampleIndices.length, node.impurity)) {
            return node; // Return leaf node
        }
        
        // Find best split
        Split bestSplit = findBestSplit(X, y, sampleIndices);
        
        if (bestSplit == null || bestSplit.impurityDecrease < minImpurityDecrease) {
            return node; // Return leaf node
        }
        
        // Apply best split
        node.featureIndex = bestSplit.featureIndex;
        node.threshold = bestSplit.threshold;
        
        // Split samples
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : sampleIndices) {
            if (X[idx][bestSplit.featureIndex] <= bestSplit.threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        // Recursively build left and right subtrees
        if (!leftIndices.isEmpty()) {
            int[] leftArray = leftIndices.stream().mapToInt(Integer::intValue).toArray();
            node.left = buildTree(X, y, leftArray, depth + 1);
        }
        
        if (!rightIndices.isEmpty()) {
            int[] rightArray = rightIndices.stream().mapToInt(Integer::intValue).toArray();
            node.right = buildTree(X, y, rightArray, depth + 1);
        }
        
        return node;
    }
    
    /**
     * Find the best split for the given samples.
     */
    private Split findBestSplit(double[][] X, double[] y, int[] sampleIndices) {
        Split bestSplit = null;
        double bestImpurityDecrease = 0.0;
        
        // Select random subset of features if max_features < total features
        int[] featureIndices = selectFeatures(X[0].length);
        
        for (int featureIdx : featureIndices) {
            // Get unique values for this feature
            Set<Double> uniqueValues = new HashSet<>();
            for (int idx : sampleIndices) {
                uniqueValues.add(X[idx][featureIdx]);
            }
            
            List<Double> sortedValues = new ArrayList<>(uniqueValues);
            Collections.sort(sortedValues);
            
            // Try splits between consecutive unique values
            for (int i = 0; i < sortedValues.size() - 1; i++) {
                double threshold = (sortedValues.get(i) + sortedValues.get(i + 1)) / 2.0;
                
                Split split = evaluateSplit(X, y, sampleIndices, featureIdx, threshold);
                
                if (split != null && split.impurityDecrease > bestImpurityDecrease) {
                    bestImpurityDecrease = split.impurityDecrease;
                    bestSplit = split;
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * Evaluate a potential split.
     */
    private Split evaluateSplit(double[][] X, double[] y, int[] sampleIndices, int featureIndex, double threshold) {
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : sampleIndices) {
            if (X[idx][featureIndex] <= threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        // Check minimum samples constraints
        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return null;
        }
        
        // Calculate impurity decrease
        double parentImpurity = isClassification ? 
            calculateImpurity(y, sampleIndices, criterion) :
            calculateMse(y, sampleIndices, calculateMean(y, sampleIndices));
            
        double leftImpurity = isClassification ?
            calculateImpurity(y, leftIndices.stream().mapToInt(Integer::intValue).toArray(), criterion) :
            calculateMse(y, leftIndices.stream().mapToInt(Integer::intValue).toArray(), 
                        calculateMean(y, leftIndices.stream().mapToInt(Integer::intValue).toArray()));
            
        double rightImpurity = isClassification ?
            calculateImpurity(y, rightIndices.stream().mapToInt(Integer::intValue).toArray(), criterion) :
            calculateMse(y, rightIndices.stream().mapToInt(Integer::intValue).toArray(),
                        calculateMean(y, rightIndices.stream().mapToInt(Integer::intValue).toArray()));
        
        double weightedImpurity = (leftIndices.size() * leftImpurity + rightIndices.size() * rightImpurity) / sampleIndices.length;
        double impurityDecrease = parentImpurity - weightedImpurity;
        
        return new Split(featureIndex, threshold, impurityDecrease);
    }
    
    // Helper methods
    
    private boolean isClassificationProblem(double[] y) {
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        return uniqueValues.size() <= Math.sqrt(y.length); // Heuristic: if unique values <= sqrt(n), treat as classification
    }
    
    private boolean shouldStop(int depth, int samples, double impurity) {
        return depth >= maxDepth || 
               samples < minSamplesSplit || 
               samples < 2 * minSamplesLeaf ||
               impurity < 1e-7;
    }
    
    private int[] selectFeatures(int totalFeatures) {
        if (maxFeatures >= totalFeatures) {
            return IntStream.range(0, totalFeatures).toArray();
        }
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalFeatures; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        return indices.stream().limit(maxFeatures).mapToInt(Integer::intValue).toArray();
    }
    
    private double[] calculateClassProbabilities(double[] y, int[] indices) {
        Map<Double, Integer> classCounts = new HashMap<>();
        for (int idx : indices) {
            classCounts.merge(y[idx], 1, Integer::sum);
        }
        
        double[] probabilities = new double[classes.length];
        for (int i = 0; i < classes.length; i++) {
            probabilities[i] = classCounts.getOrDefault(classes[i], 0) / (double) indices.length;
        }
        
        return probabilities;
    }
    
    private double calculateMean(double[] y, int[] indices) {
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        return sum / indices.length;
    }
    
    private double calculateImpurity(double[] y, int[] indices, String criterion) {
        if (criterion.equals("gini")) {
            return calculateGini(y, indices);
        } else if (criterion.equals("entropy")) {
            return calculateEntropy(y, indices);
        }
        return 0.0;
    }
    
    private double calculateGini(double[] y, int[] indices) {
        Map<Double, Integer> classCounts = new HashMap<>();
        for (int idx : indices) {
            classCounts.merge(y[idx], 1, Integer::sum);
        }
        
        double gini = 1.0;
        for (int count : classCounts.values()) {
            double probability = count / (double) indices.length;
            gini -= probability * probability;
        }
        
        return gini;
    }
    
    private double calculateEntropy(double[] y, int[] indices) {
        Map<Double, Integer> classCounts = new HashMap<>();
        for (int idx : indices) {
            classCounts.merge(y[idx], 1, Integer::sum);
        }
        
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = count / (double) indices.length;
                entropy -= probability * (Math.log(probability) / Math.log(2)); // log2(x) = ln(x) / ln(2)
            }
        }
        
        return entropy;
    }
    
    private double calculateMse(double[] y, int[] indices, double mean) {
        double mse = 0.0;
        for (int idx : indices) {
            double diff = y[idx] - mean;
            mse += diff * diff;
        }
        return mse / indices.length;
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
    
    private double predictSample(double[] sample) {
        TreeNode node = root;
        
        while (!node.isLeaf()) {
            if (sample[node.featureIndex] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        
        return node.prediction;
    }
    
    private double[] predictProbaSample(double[] sample) {
        TreeNode node = root;
        
        while (!node.isLeaf()) {
            if (sample[node.featureIndex] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        
        return Arrays.copyOf(node.value, node.value.length);
    }
    
    private int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    // Getters and setters
    
    public String getCriterion() { return criterion; }
    public DecisionTree setCriterion(String criterion) {
        this.criterion = criterion;
        params.put("criterion", criterion);
        return this;
    }
    
    public int getMaxDepth() { return maxDepth; }
    public DecisionTree setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth;
        params.put("max_depth", maxDepth);
        return this;
    }
    
    public int getMinSamplesSplit() { return minSamplesSplit; }
    public DecisionTree setMinSamplesSplit(int minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
        params.put("min_samples_split", minSamplesSplit);
        return this;
    }
    
    public int getMinSamplesLeaf() { return minSamplesLeaf; }
    public DecisionTree setMinSamplesLeaf(int minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
        params.put("min_samples_leaf", minSamplesLeaf);
        return this;
    }
    
    public double getMinImpurityDecrease() { return minImpurityDecrease; }
    public DecisionTree setMinImpurityDecrease(double minImpurityDecrease) {
        this.minImpurityDecrease = minImpurityDecrease;
        params.put("min_impurity_decrease", minImpurityDecrease);
        return this;
    }
    
    public int getMaxFeatures() { return maxFeatures; }
    public DecisionTree setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
        params.put("max_features", maxFeatures);
        return this;
    }
    
    public int getRandomState() { return randomState; }
    public DecisionTree setRandomState(int randomState) {
        this.randomState = randomState;
        this.random = new Random(randomState);
        params.put("random_state", randomState);
        return this;
    }
    
    /**
     * Get the root node of the tree (for debugging/visualization).
     */
    public TreeNode getTree() {
        return root;
    }
    
    /**
     * Internal class for representing a split during tree construction.
     */
    private static class Split {
        final int featureIndex;
        final double threshold;
        final double impurityDecrease;
        
        Split(int featureIndex, double threshold, double impurityDecrease) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.impurityDecrease = impurityDecrease;
        }
    }
}
