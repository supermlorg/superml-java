package org.superml.cluster;

import org.superml.core.BaseEstimator;
import org.superml.core.UnsupervisedLearner;
import java.util.Arrays;
import java.util.Random;

/**
 * K-Means clustering algorithm.
 * Similar to sklearn.cluster.KMeans
 */
public class KMeans extends BaseEstimator implements UnsupervisedLearner {
    
    private double[][] clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean fitted = false;
    
    // Hyperparameters
    private int nClusters = 8;
    private int maxIter = 300;
    private double tolerance = 1e-4;
    private String initMethod = "k-means++";
    private int randomState = 42;
    private int nInit = 10;
    
    public KMeans() {
        params.put("n_clusters", nClusters);
        params.put("max_iter", maxIter);
        params.put("tol", tolerance);
        params.put("init", initMethod);
        params.put("random_state", randomState);
        params.put("n_init", nInit);
    }
    
    public KMeans(int nClusters) {
        this();
        this.nClusters = nClusters;
        params.put("n_clusters", nClusters);
    }
    
    public KMeans(int nClusters, int maxIter, int randomState) {
        this();
        this.nClusters = nClusters;
        this.maxIter = maxIter;
        this.randomState = randomState;
        params.put("n_clusters", nClusters);
        params.put("max_iter", maxIter);
        params.put("random_state", randomState);
    }
    
    @Override
    public KMeans fit(double[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        Random random = new Random(randomState);
        double bestInertia = Double.POSITIVE_INFINITY;
        double[][] bestCenters = null;
        int[] bestLabels = null;
        
        // Run k-means multiple times and keep the best result
        for (int run = 0; run < nInit; run++) {
            // Initialize cluster centers
            double[][] centers = initializeCenters(X, random);
            int[] currentLabels = new int[nSamples];
            
            // K-means iterations
            for (int iter = 0; iter < maxIter; iter++) {
                // Assign points to nearest clusters
                boolean changed = assignClusters(X, centers, currentLabels);
                
                // Update cluster centers
                double[][] newCenters = updateCenters(X, currentLabels, nFeatures);
                
                // Check for convergence
                if (!changed || centersConverged(centers, newCenters)) {
                    break;
                }
                
                centers = newCenters;
            }
            
            // Calculate inertia for this run
            double currentInertia = calculateInertia(X, centers, currentLabels);
            
            // Keep the best result
            if (currentInertia < bestInertia) {
                bestInertia = currentInertia;
                bestCenters = centers;
                bestLabels = currentLabels.clone();
            }
        }
        
        this.clusterCenters = bestCenters;
        this.labels = bestLabels;
        this.inertia = bestInertia;
        this.fitted = true;
        
        return this;
    }
    
    @Override
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("KMeans must be fitted before transforming data");
        }
        
        // Transform to distances from cluster centers
        int nSamples = X.length;
        double[][] distances = new double[nSamples][nClusters];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nClusters; j++) {
                distances[i][j] = euclideanDistance(X[i], clusterCenters[j]);
            }
        }
        
        return distances;
    }
    
    /**
     * Predict the closest cluster for each sample.
     * @param X samples to predict
     * @return cluster labels
     */
    public int[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("KMeans must be fitted before making predictions");
        }
        
        int[] predictions = new int[X.length];
        assignClusters(X, clusterCenters, predictions);
        return predictions;
    }
    
    /**
     * Fit and predict in one step.
     * @param X data to cluster
     * @return cluster labels
     */
    public int[] fitPredict(double[][] X) {
        fit(X);
        return labels.clone();
    }
    
    private double[][] initializeCenters(double[][] X, Random random) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] centers = new double[nClusters][nFeatures];
        
        if ("k-means++".equals(initMethod)) {
            // K-means++ initialization
            // Choose first center randomly
            int firstCenter = random.nextInt(nSamples);
            centers[0] = X[firstCenter].clone();
            
            // Choose remaining centers
            for (int c = 1; c < nClusters; c++) {
                double[] distances = new double[nSamples];
                double totalDistance = 0.0;
                
                // Calculate distances to nearest center
                for (int i = 0; i < nSamples; i++) {
                    double minDist = Double.POSITIVE_INFINITY;
                    for (int j = 0; j < c; j++) {
                        double dist = euclideanDistanceSquared(X[i], centers[j]);
                        minDist = Math.min(minDist, dist);
                    }
                    distances[i] = minDist;
                    totalDistance += minDist;
                }
                
                // Choose next center with probability proportional to distance
                double target = random.nextDouble() * totalDistance;
                double cumSum = 0.0;
                
                for (int i = 0; i < nSamples; i++) {
                    cumSum += distances[i];
                    if (cumSum >= target) {
                        centers[c] = X[i].clone();
                        break;
                    }
                }
            }
        } else {
            // Random initialization
            for (int i = 0; i < nClusters; i++) {
                int randomIndex = random.nextInt(nSamples);
                centers[i] = X[randomIndex].clone();
            }
        }
        
        return centers;
    }
    
    private boolean assignClusters(double[][] X, double[][] centers, int[] labels) {
        boolean changed = false;
        
        for (int i = 0; i < X.length; i++) {
            int nearestCluster = 0;
            double minDistance = euclideanDistance(X[i], centers[0]);
            
            for (int j = 1; j < nClusters; j++) {
                double distance = euclideanDistance(X[i], centers[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCluster = j;
                }
            }
            
            if (labels[i] != nearestCluster) {
                changed = true;
                labels[i] = nearestCluster;
            }
        }
        
        return changed;
    }
    
    private double[][] updateCenters(double[][] X, int[] labels, int nFeatures) {
        double[][] newCenters = new double[nClusters][nFeatures];
        int[] counts = new int[nClusters];
        
        // Sum points in each cluster
        for (int i = 0; i < X.length; i++) {
            int cluster = labels[i];
            counts[cluster]++;
            for (int j = 0; j < nFeatures; j++) {
                newCenters[cluster][j] += X[i][j];
            }
        }
        
        // Calculate means
        for (int i = 0; i < nClusters; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < nFeatures; j++) {
                    newCenters[i][j] /= counts[i];
                }
            }
        }
        
        return newCenters;
    }
    
    private boolean centersConverged(double[][] oldCenters, double[][] newCenters) {
        for (int i = 0; i < nClusters; i++) {
            double distance = euclideanDistance(oldCenters[i], newCenters[i]);
            if (distance > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    private double calculateInertia(double[][] X, double[][] centers, int[] labels) {
        double totalInertia = 0.0;
        
        for (int i = 0; i < X.length; i++) {
            int cluster = labels[i];
            double distance = euclideanDistanceSquared(X[i], centers[cluster]);
            totalInertia += distance;
        }
        
        return totalInertia;
    }
    
    private double euclideanDistance(double[] point1, double[] point2) {
        return Math.sqrt(euclideanDistanceSquared(point1, point2));
    }
    
    private double euclideanDistanceSquared(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    // Getters
    public double[][] getClusterCenters() {
        if (!fitted) {
            throw new IllegalStateException("KMeans must be fitted before accessing cluster centers");
        }
        return Arrays.stream(clusterCenters)
                     .map(center -> center.clone())
                     .toArray(double[][]::new);
    }
    
    public int[] getLabels() {
        if (!fitted) {
            throw new IllegalStateException("KMeans must be fitted before accessing labels");
        }
        return labels.clone();
    }
    
    public double getInertia() {
        if (!fitted) {
            throw new IllegalStateException("KMeans must be fitted before accessing inertia");
        }
        return inertia;
    }
    
    // Setters
    public KMeans setNClusters(int nClusters) {
        this.nClusters = nClusters;
        params.put("n_clusters", nClusters);
        return this;
    }
    
    public KMeans setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public KMeans setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public KMeans setRandomState(int randomState) {
        this.randomState = randomState;
        params.put("random_state", randomState);
        return this;
    }
    
    public KMeans setInitMethod(String initMethod) {
        this.initMethod = initMethod;
        params.put("init", initMethod);
        return this;
    }
    
    public KMeans setNInit(int nInit) {
        this.nInit = nInit;
        params.put("n_init", nInit);
        return this;
    }
}
