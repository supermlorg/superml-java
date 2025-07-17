/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.visualization;

import org.superml.tree.XGBoost;
import org.superml.metrics.XGBoostMetrics;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

/**
 * XGBoost visualization utilities for feature importance, learning curves, and model analysis
 * 
 * Provides comprehensive visualization capabilities including:
 * - Feature importance plots (weight, gain, cover)
 * - Learning curves and training progress
 * - Tree structure visualization
 * - Hyperparameter optimization plots
 * - Model performance comparison
 * - Overfitting detection plots
 * 
 * Outputs data in formats compatible with popular plotting libraries
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoostVisualization {
    
    /**
     * Generate feature importance plot data
     */
    public static FeatureImportancePlot plotFeatureImportance(XGBoost model, String[] featureNames, 
                                                            String importanceType) {
        if (!model.isFitted()) {
            throw new IllegalStateException("Model must be fitted before plotting feature importance");
        }
        
        Map<String, double[]> importanceStats = model.getFeatureImportanceStats();
        double[] importance = importanceStats.get(importanceType.toLowerCase());
        
        if (importance == null) {
            throw new IllegalArgumentException("Invalid importance type: " + importanceType + 
                ". Valid types: weight, gain, cover");
        }
        
        FeatureImportancePlot plot = new FeatureImportancePlot();
        plot.importanceType = importanceType;
        plot.featureNames = featureNames != null ? featureNames : generateFeatureNames(importance.length);
        plot.importance = importance;
        
        // Sort features by importance
        Integer[] indices = IntStream.range(0, importance.length)
            .boxed()
            .toArray(Integer[]::new);
        Arrays.sort(indices, (i, j) -> Double.compare(importance[j], importance[i]));
        
        plot.sortedIndices = indices;
        plot.sortedFeatureNames = Arrays.stream(indices)
            .map(i -> plot.featureNames[i])
            .toArray(String[]::new);
        plot.sortedImportance = Arrays.stream(indices)
            .mapToDouble(i -> importance[i])
            .toArray();
        
        return plot;
    }
    
    /**
     * Generate learning curve plot data
     */
    public static LearningCurvePlot plotLearningCurves(XGBoost model) {
        if (!model.isFitted()) {
            throw new IllegalStateException("Model must be fitted before plotting learning curves");
        }
        
        Map<String, List<Double>> evalResults = model.getEvalResults();
        
        LearningCurvePlot plot = new LearningCurvePlot();
        plot.evalResults = evalResults;
        
        // Extract training and validation curves
        for (Map.Entry<String, List<Double>> entry : evalResults.entrySet()) {
            String metricName = entry.getKey();
            List<Double> values = entry.getValue();
            
            if (metricName.startsWith("train")) {
                plot.trainMetrics.put(metricName, values);
            } else if (metricName.startsWith("valid")) {
                plot.validMetrics.put(metricName, values);
            }
        }
        
        // Generate epoch numbers
        if (!evalResults.isEmpty()) {
            int nEpochs = evalResults.values().iterator().next().size();
            plot.epochs = IntStream.range(1, nEpochs + 1).toArray();
        }
        
        // Detect overfitting point
        plot.overfittingPoint = detectOverfittingPoint(plot.trainMetrics, plot.validMetrics);
        
        return plot;
    }
    
    /**
     * Generate hyperparameter optimization plot data
     */
    public static HyperparameterPlot plotHyperparameterOptimization(List<HyperparameterResult> results,
                                                                  String xParam, String yParam) {
        HyperparameterPlot plot = new HyperparameterPlot();
        plot.xParam = xParam;
        plot.yParam = yParam;
        plot.results = results;
        
        // Extract parameter values and scores
        plot.xValues = results.stream()
            .mapToDouble(r -> r.getParameterValue(xParam))
            .toArray();
        
        plot.yValues = results.stream()
            .mapToDouble(r -> r.getParameterValue(yParam))
            .toArray();
        
        plot.scores = results.stream()
            .mapToDouble(HyperparameterResult::getScore)
            .toArray();
        
        // Find best parameters
        int bestIndex = IntStream.range(0, plot.scores.length)
            .reduce((i, j) -> plot.scores[i] > plot.scores[j] ? i : j)
            .orElse(0);
        
        plot.bestX = plot.xValues[bestIndex];
        plot.bestY = plot.yValues[bestIndex];
        plot.bestScore = plot.scores[bestIndex];
        
        return plot;
    }
    
    /**
     * Generate model comparison plot data
     */
    public static ModelComparisonPlot plotModelComparison(List<ModelResult> modelResults) {
        ModelComparisonPlot plot = new ModelComparisonPlot();
        plot.modelResults = modelResults;
        
        plot.modelNames = modelResults.stream()
            .map(ModelResult::getModelName)
            .toArray(String[]::new);
        
        plot.scores = modelResults.stream()
            .mapToDouble(ModelResult::getScore)
            .toArray();
        
        plot.trainTimes = modelResults.stream()
            .mapToDouble(ModelResult::getTrainTime)
            .toArray();
        
        plot.predictTimes = modelResults.stream()
            .mapToDouble(ModelResult::getPredictTime)
            .toArray();
        
        // Calculate normalized scores for radar plot
        double maxScore = Arrays.stream(plot.scores).max().orElse(1.0);
        double maxTrainTime = Arrays.stream(plot.trainTimes).max().orElse(1.0);
        double maxPredictTime = Arrays.stream(plot.predictTimes).max().orElse(1.0);
        
        plot.normalizedScores = Arrays.stream(plot.scores)
            .map(s -> s / maxScore)
            .toArray();
        
        plot.normalizedTrainTimes = Arrays.stream(plot.trainTimes)
            .map(t -> 1.0 - (t / maxTrainTime)) // Invert for radar plot (higher is better)
            .toArray();
        
        plot.normalizedPredictTimes = Arrays.stream(plot.predictTimes)
            .map(t -> 1.0 - (t / maxPredictTime)) // Invert for radar plot (higher is better)
            .toArray();
        
        return plot;
    }
    
    /**
     * Generate residuals plot data for regression models
     */
    public static ResidualsPlot plotResiduals(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        ResidualsPlot plot = new ResidualsPlot();
        plot.yTrue = yTrue;
        plot.yPred = yPred;
        
        // Calculate residuals
        plot.residuals = new double[yTrue.length];
        for (int i = 0; i < yTrue.length; i++) {
            plot.residuals[i] = yTrue[i] - yPred[i];
        }
        
        // Calculate statistics
        plot.meanResidual = Arrays.stream(plot.residuals).average().orElse(0.0);
        plot.stdResidual = calculateStandardDeviation(plot.residuals);
        
        // Detect outliers (points beyond 2 standard deviations)
        List<Integer> outlierIndices = new ArrayList<>();
        for (int i = 0; i < plot.residuals.length; i++) {
            if (Math.abs(plot.residuals[i] - plot.meanResidual) > 2 * plot.stdResidual) {
                outlierIndices.add(i);
            }
        }
        plot.outlierIndices = outlierIndices.stream().mapToInt(Integer::intValue).toArray();
        
        return plot;
    }
    
    /**
     * Generate ROC curve data for binary classification
     */
    public static ROCCurvePlot plotROCCurve(double[] yTrue, double[] yScores) {
        if (yTrue.length != yScores.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Sort by scores descending
        List<ROCPoint> points = new ArrayList<>();
        for (int i = 0; i < yTrue.length; i++) {
            points.add(new ROCPoint(yTrue[i], yScores[i]));
        }
        points.sort((a, b) -> Double.compare(b.score, a.score));
        
        int totalPositives = (int) Arrays.stream(yTrue).sum();
        int totalNegatives = yTrue.length - totalPositives;
        
        List<Double> fprList = new ArrayList<>();
        List<Double> tprList = new ArrayList<>();
        
        int truePositives = 0;
        int falsePositives = 0;
        
        fprList.add(0.0);
        tprList.add(0.0);
        
        for (ROCPoint point : points) {
            if (point.label == 1.0) {
                truePositives++;
            } else {
                falsePositives++;
            }
            
            double fpr = (double) falsePositives / totalNegatives;
            double tpr = (double) truePositives / totalPositives;
            
            fprList.add(fpr);
            tprList.add(tpr);
        }
        
        ROCCurvePlot plot = new ROCCurvePlot();
        plot.fpr = fprList.stream().mapToDouble(Double::doubleValue).toArray();
        plot.tpr = tprList.stream().mapToDouble(Double::doubleValue).toArray();
        
        // Calculate AUC using trapezoidal rule
        plot.auc = 0.0;
        for (int i = 1; i < plot.fpr.length; i++) {
            plot.auc += (plot.fpr[i] - plot.fpr[i-1]) * (plot.tpr[i] + plot.tpr[i-1]) / 2.0;
        }
        
        return plot;
    }
    
    /**
     * Export plot data to CSV format
     */
    public static void exportToCSV(PlotData plotData, String filename) throws IOException {
        try (FileWriter writer = new FileWriter(filename)) {
            plotData.writeCSV(writer);
        }
    }
    
    /**
     * Export plot data to JSON format for web visualization
     */
    public static String exportToJSON(PlotData plotData) {
        return plotData.toJSON();
    }
    
    /**
     * Generate Python plotting code for matplotlib
     */
    public static String generatePythonCode(PlotData plotData) {
        return plotData.toPythonCode();
    }
    
    /**
     * Generate JavaScript plotting code for D3.js
     */
    public static String generateJavaScriptCode(PlotData plotData) {
        return plotData.toJavaScriptCode();
    }
    
    // Helper methods
    
    private static String[] generateFeatureNames(int nFeatures) {
        String[] names = new String[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            names[i] = "feature_" + i;
        }
        return names;
    }
    
    private static int detectOverfittingPoint(Map<String, List<Double>> trainMetrics,
                                            Map<String, List<Double>> validMetrics) {
        // Simple overfitting detection: find where validation score stops improving
        if (trainMetrics.isEmpty() || validMetrics.isEmpty()) {
            return -1;
        }
        
        List<Double> validScores = validMetrics.values().iterator().next();
        if (validScores.size() < 10) {
            return -1;
        }
        
        double bestScore = validScores.get(0);
        int bestEpoch = 0;
        int patienceCounter = 0;
        int patience = 5;
        
        for (int i = 1; i < validScores.size(); i++) {
            if (validScores.get(i) < bestScore) { // Assuming lower is better (loss)
                bestScore = validScores.get(i);
                bestEpoch = i;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= patience) {
                    return bestEpoch;
                }
            }
        }
        
        return -1; // No overfitting detected
    }
    
    private static double calculateStandardDeviation(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values)
            .map(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    // Data classes and interfaces
    
    public interface PlotData {
        void writeCSV(FileWriter writer) throws IOException;
        String toJSON();
        String toPythonCode();
        String toJavaScriptCode();
    }
    
    public static class FeatureImportancePlot implements PlotData {
        public String importanceType;
        public String[] featureNames;
        public double[] importance;
        public Integer[] sortedIndices;
        public String[] sortedFeatureNames;
        public double[] sortedImportance;
        
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write("feature,importance\n");
            for (int i = 0; i < Math.min(20, sortedFeatureNames.length); i++) { // Top 20 features
                writer.write(String.format("%s,%.6f\n", sortedFeatureNames[i], sortedImportance[i]));
            }
        }
        
        @Override
        public String toJSON() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"importanceType\": \"").append(importanceType).append("\",\n");
            sb.append("  \"features\": [\n");
            
            int topN = Math.min(20, sortedFeatureNames.length);
            for (int i = 0; i < topN; i++) {
                sb.append("    {\"name\": \"").append(sortedFeatureNames[i]).append("\", ");
                sb.append("\"importance\": ").append(sortedImportance[i]).append("}");
                if (i < topN - 1) sb.append(",");
                sb.append("\n");
            }
            
            sb.append("  ]\n}");
            return sb.toString();
        }
        
        @Override
        public String toPythonCode() {
            StringBuilder sb = new StringBuilder();
            sb.append("import matplotlib.pyplot as plt\n");
            sb.append("import numpy as np\n\n");
            
            sb.append("# Feature importance data\n");
            sb.append("features = [");
            int topN = Math.min(20, sortedFeatureNames.length);
            for (int i = 0; i < topN; i++) {
                sb.append("\"").append(sortedFeatureNames[i]).append("\"");
                if (i < topN - 1) sb.append(", ");
            }
            sb.append("]\n");
            
            sb.append("importance = [");
            for (int i = 0; i < topN; i++) {
                sb.append(sortedImportance[i]);
                if (i < topN - 1) sb.append(", ");
            }
            sb.append("]\n\n");
            
            sb.append("# Create horizontal bar plot\n");
            sb.append("plt.figure(figsize=(10, 8))\n");
            sb.append("plt.barh(range(len(features)), importance)\n");
            sb.append("plt.yticks(range(len(features)), features)\n");
            sb.append("plt.xlabel('").append(importanceType.substring(0, 1).toUpperCase())
              .append(importanceType.substring(1)).append(" Importance')\n");
            sb.append("plt.title('XGBoost Feature Importance')\n");
            sb.append("plt.gca().invert_yaxis()\n");
            sb.append("plt.tight_layout()\n");
            sb.append("plt.show()\n");
            
            return sb.toString();
        }
        
        @Override
        public String toJavaScriptCode() {
            StringBuilder sb = new StringBuilder();
            sb.append("// D3.js feature importance plot\n");
            sb.append("const data = [\n");
            
            int topN = Math.min(20, sortedFeatureNames.length);
            for (int i = 0; i < topN; i++) {
                sb.append("  {name: \"").append(sortedFeatureNames[i]).append("\", ");
                sb.append("value: ").append(sortedImportance[i]).append("}");
                if (i < topN - 1) sb.append(",");
                sb.append("\n");
            }
            
            sb.append("];\n\n");
            sb.append("// Create SVG and draw horizontal bar chart\n");
            sb.append("// (D3.js implementation would go here)\n");
            
            return sb.toString();
        }
    }
    
    public static class LearningCurvePlot implements PlotData {
        public Map<String, List<Double>> evalResults = new HashMap<>();
        public Map<String, List<Double>> trainMetrics = new HashMap<>();
        public Map<String, List<Double>> validMetrics = new HashMap<>();
        public int[] epochs;
        public int overfittingPoint = -1;
        
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write("epoch");
            for (String metric : evalResults.keySet()) {
                writer.write("," + metric);
            }
            writer.write("\n");
            
            for (int i = 0; i < epochs.length; i++) {
                writer.write(String.valueOf(epochs[i]));
                for (String metric : evalResults.keySet()) {
                    List<Double> values = evalResults.get(metric);
                    if (i < values.size()) {
                        writer.write("," + values.get(i));
                    } else {
                        writer.write(",");
                    }
                }
                writer.write("\n");
            }
        }
        
        @Override
        public String toJSON() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"epochs\": [");
            for (int i = 0; i < epochs.length; i++) {
                sb.append(epochs[i]);
                if (i < epochs.length - 1) sb.append(", ");
            }
            sb.append("],\n");
            
            sb.append("  \"metrics\": {\n");
            int metricCount = 0;
            for (Map.Entry<String, List<Double>> entry : evalResults.entrySet()) {
                sb.append("    \"").append(entry.getKey()).append("\": [");
                List<Double> values = entry.getValue();
                for (int i = 0; i < values.size(); i++) {
                    sb.append(values.get(i));
                    if (i < values.size() - 1) sb.append(", ");
                }
                sb.append("]");
                if (++metricCount < evalResults.size()) sb.append(",");
                sb.append("\n");
            }
            sb.append("  },\n");
            
            sb.append("  \"overfittingPoint\": ").append(overfittingPoint).append("\n");
            sb.append("}");
            
            return sb.toString();
        }
        
        @Override
        public String toPythonCode() {
            StringBuilder sb = new StringBuilder();
            sb.append("import matplotlib.pyplot as plt\n");
            sb.append("import numpy as np\n\n");
            
            sb.append("epochs = np.array([");
            for (int i = 0; i < epochs.length; i++) {
                sb.append(epochs[i]);
                if (i < epochs.length - 1) sb.append(", ");
            }
            sb.append("])\n\n");
            
            for (Map.Entry<String, List<Double>> entry : evalResults.entrySet()) {
                String metricName = entry.getKey().replace("-", "_");
                sb.append(metricName).append(" = [");
                List<Double> values = entry.getValue();
                for (int i = 0; i < values.size(); i++) {
                    sb.append(values.get(i));
                    if (i < values.size() - 1) sb.append(", ");
                }
                sb.append("]\n");
            }
            
            sb.append("\nplt.figure(figsize=(12, 8))\n");
            for (String metric : evalResults.keySet()) {
                String metricName = metric.replace("-", "_");
                String label = metric.replace("-", " ").replace("_", " ");
                sb.append("plt.plot(epochs, ").append(metricName).append(", label='").append(label).append("')\n");
            }
            
            if (overfittingPoint > 0) {
                sb.append("plt.axvline(x=").append(overfittingPoint).append(", color='red', linestyle='--', ");
                sb.append("label='Overfitting Point')\n");
            }
            
            sb.append("plt.xlabel('Epoch')\n");
            sb.append("plt.ylabel('Score')\n");
            sb.append("plt.title('XGBoost Learning Curves')\n");
            sb.append("plt.legend()\n");
            sb.append("plt.grid(True)\n");
            sb.append("plt.show()\n");
            
            return sb.toString();
        }
        
        @Override
        public String toJavaScriptCode() {
            return "// D3.js learning curve implementation\n// (Implementation would go here)";
        }
    }
    
    // Additional plot classes
    public static class HyperparameterPlot implements PlotData {
        public String xParam;
        public String yParam;
        public List<HyperparameterResult> results;
        public double[] xValues;
        public double[] yValues;
        public double[] scores;
        public double bestX;
        public double bestY;
        public double bestScore;
        
        // Implementation methods...
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write(xParam + "," + yParam + ",score\n");
            for (int i = 0; i < xValues.length; i++) {
                writer.write(String.format("%.6f,%.6f,%.6f\n", xValues[i], yValues[i], scores[i]));
            }
        }
        
        @Override
        public String toJSON() { return "{}"; } // Simplified
        @Override
        public String toPythonCode() { return "# Hyperparameter plot code"; }
        @Override
        public String toJavaScriptCode() { return "// Hyperparameter plot code"; }
    }
    
    public static class ModelComparisonPlot implements PlotData {
        public List<ModelResult> modelResults;
        public String[] modelNames;
        public double[] scores;
        public double[] trainTimes;
        public double[] predictTimes;
        public double[] normalizedScores;
        public double[] normalizedTrainTimes;
        public double[] normalizedPredictTimes;
        
        // Implementation methods...
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write("model,score,train_time,predict_time\n");
            for (int i = 0; i < modelNames.length; i++) {
                writer.write(String.format("%s,%.6f,%.6f,%.6f\n", 
                    modelNames[i], scores[i], trainTimes[i], predictTimes[i]));
            }
        }
        
        @Override
        public String toJSON() { return "{}"; }
        @Override
        public String toPythonCode() { return "# Model comparison plot code"; }
        @Override
        public String toJavaScriptCode() { return "// Model comparison plot code"; }
    }
    
    public static class ResidualsPlot implements PlotData {
        public double[] yTrue;
        public double[] yPred;
        public double[] residuals;
        public double meanResidual;
        public double stdResidual;
        public int[] outlierIndices;
        
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write("y_true,y_pred,residual\n");
            for (int i = 0; i < yTrue.length; i++) {
                writer.write(String.format("%.6f,%.6f,%.6f\n", yTrue[i], yPred[i], residuals[i]));
            }
        }
        
        @Override
        public String toJSON() { return "{}"; }
        @Override
        public String toPythonCode() { return "# Residuals plot code"; }
        @Override
        public String toJavaScriptCode() { return "// Residuals plot code"; }
    }
    
    public static class ROCCurvePlot implements PlotData {
        public double[] fpr;
        public double[] tpr;
        public double auc;
        
        @Override
        public void writeCSV(FileWriter writer) throws IOException {
            writer.write("fpr,tpr\n");
            for (int i = 0; i < fpr.length; i++) {
                writer.write(String.format("%.6f,%.6f\n", fpr[i], tpr[i]));
            }
        }
        
        @Override
        public String toJSON() { return "{}"; }
        @Override
        public String toPythonCode() { return "# ROC curve plot code"; }
        @Override
        public String toJavaScriptCode() { return "// ROC curve plot code"; }
    }
    
    // Helper classes
    public static class HyperparameterResult {
        private Map<String, Double> parameters;
        private double score;
        
        public HyperparameterResult(Map<String, Double> parameters, double score) {
            this.parameters = parameters;
            this.score = score;
        }
        
        public double getParameterValue(String param) {
            return parameters.getOrDefault(param, 0.0);
        }
        
        public double getScore() {
            return score;
        }
    }
    
    public static class ModelResult {
        private String modelName;
        private double score;
        private double trainTime;
        private double predictTime;
        
        public ModelResult(String modelName, double score, double trainTime, double predictTime) {
            this.modelName = modelName;
            this.score = score;
            this.trainTime = trainTime;
            this.predictTime = predictTime;
        }
        
        public String getModelName() { return modelName; }
        public double getScore() { return score; }
        public double getTrainTime() { return trainTime; }
        public double getPredictTime() { return predictTime; }
    }
    
    private static class ROCPoint {
        final double label;
        final double score;
        
        ROCPoint(double label, double score) {
            this.label = label;
            this.score = score;
        }
    }
}
