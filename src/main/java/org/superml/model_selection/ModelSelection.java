package org.superml.model_selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Model selection utilities.
 * Similar to sklearn.model_selection
 */
public class ModelSelection {
    
    /**
     * Result class for train-test split.
     */
    public static class TrainTestSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;
        
        public TrainTestSplit(double[][] XTrain, double[][] XTest, 
                             double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
    
    /**
     * Split arrays into random train and test subsets.
     * @param X features
     * @param y targets
     * @param testSize proportion of the dataset to include in the test split
     * @param randomState random seed for reproducible output
     * @return TrainTestSplit object
     */
    public static TrainTestSplit trainTestSplit(double[][] X, double[] y, 
                                               double testSize, int randomState) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int nSamples = X.length;
        int nTest = (int) Math.round(nSamples * testSize);
        int nTrain = nSamples - nTest;
        
        // Create indices and shuffle them
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        
        Random random = new Random(randomState);
        Collections.shuffle(indices, random);
        
        // Split indices
        List<Integer> trainIndices = indices.subList(0, nTrain);
        List<Integer> testIndices = indices.subList(nTrain, nSamples);
        
        // Create train and test arrays
        double[][] XTrain = new double[nTrain][];
        double[] yTrain = new double[nTrain];
        double[][] XTest = new double[nTest][];
        double[] yTest = new double[nTest];
        
        for (int i = 0; i < nTrain; i++) {
            int idx = trainIndices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < nTest; i++) {
            int idx = testIndices.get(i);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }
        
        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
    
    /**
     * Split arrays into random train and test subsets with default random state.
     * @param X features
     * @param y targets
     * @param testSize proportion of the dataset to include in the test split
     * @return TrainTestSplit object
     */
    public static TrainTestSplit trainTestSplit(double[][] X, double[] y, double testSize) {
        return trainTestSplit(X, y, testSize, 42);
    }
    
    /**
     * K-Fold cross-validation iterator.
     */
    public static class KFold {
        private final int nSplits;
        private final boolean shuffle;
        private final int randomState;
        
        public KFold(int nSplits, boolean shuffle, int randomState) {
            this.nSplits = nSplits;
            this.shuffle = shuffle;
            this.randomState = randomState;
        }
        
        public KFold(int nSplits) {
            this(nSplits, false, 42);
        }
        
        /**
         * Generate indices to split data into training and test set.
         * @param nSamples number of samples
         * @return list of train-test index pairs
         */
        public List<int[][]> split(int nSamples) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                indices.add(i);
            }
            
            if (shuffle) {
                Random random = new Random(randomState);
                Collections.shuffle(indices, random);
            }
            
            List<int[][]> splits = new ArrayList<>();
            int foldSize = nSamples / nSplits;
            
            for (int i = 0; i < nSplits; i++) {
                int testStart = i * foldSize;
                int testEnd = (i == nSplits - 1) ? nSamples : (i + 1) * foldSize;
                
                List<Integer> trainIndices = new ArrayList<>();
                List<Integer> testIndices = new ArrayList<>();
                
                for (int j = 0; j < nSamples; j++) {
                    if (j >= testStart && j < testEnd) {
                        testIndices.add(indices.get(j));
                    } else {
                        trainIndices.add(indices.get(j));
                    }
                }
                
                int[] trainArray = trainIndices.stream().mapToInt(Integer::intValue).toArray();
                int[] testArray = testIndices.stream().mapToInt(Integer::intValue).toArray();
                
                splits.add(new int[][]{trainArray, testArray});
            }
            
            return splits;
        }
    }
    
    /**
     * Evaluate a score by cross-validation.
     * @param estimator the estimator to evaluate
     * @param X features
     * @param y targets
     * @param cv cross-validation generator
     * @return array of scores
     */
    public static double[] crossValidateScore(Object estimator, double[][] X, double[] y, KFold cv) {
        List<int[][]> splits = cv.split(X.length);
        double[] scores = new double[splits.size()];
        
        for (int i = 0; i < splits.size(); i++) {
            int[] trainIndices = splits.get(i)[0];
            int[] testIndices = splits.get(i)[1];
            
            // Extract train and test data
            double[][] XTrain = new double[trainIndices.length][];
            double[] yTrain = new double[trainIndices.length];
            double[][] XTest = new double[testIndices.length][];
            double[] yTest = new double[testIndices.length];
            
            for (int j = 0; j < trainIndices.length; j++) {
                XTrain[j] = X[trainIndices[j]].clone();
                yTrain[j] = y[trainIndices[j]];
            }
            
            for (int j = 0; j < testIndices.length; j++) {
                XTest[j] = X[testIndices[j]].clone();
                yTest[j] = y[testIndices[j]];
            }
            
            // This is a simplified version - in practice, you'd need to use reflection
            // or a common interface to call fit and score methods
            // For now, this serves as a template
            scores[i] = 0.0; // Placeholder
        }
        
        return scores;
    }
}
