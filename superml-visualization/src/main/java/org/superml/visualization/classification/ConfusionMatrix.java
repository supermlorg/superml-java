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

package org.superml.visualization.classification;

import org.superml.visualization.Visualization;

/**
 * Comprehensive Confusion Matrix implementation with visualization capabilities
 * 
 * Features:
 * - Multi-class classification support
 * - ASCII-based matrix display
 * - Per-class precision, recall, F1-score
 * - Macro and micro averages
 * - Classification report generation
 */
public class ConfusionMatrix implements Visualization {
    
    private final int[][] matrix;
    private final int numClasses;
    private final int[] actual;
    private final int[] predicted;
    private String title = "Confusion Matrix";
    private String[] classNames;
    
    /**
     * Create confusion matrix from actual and predicted labels
     * @param actual Array of actual class labels
     * @param predicted Array of predicted class labels
     * @param numClasses Number of classes
     */
    public ConfusionMatrix(int[] actual, int[] predicted, int numClasses) {
        this.actual = actual;
        this.predicted = predicted;
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.classNames = generateDefaultClassNames();
        
        buildMatrix();
    }
    
    /**
     * Create confusion matrix with custom class names
     * @param actual Array of actual class labels
     * @param predicted Array of predicted class labels
     * @param classNames Array of class names for display
     */
    public ConfusionMatrix(int[] actual, int[] predicted, String[] classNames) {
        this.actual = actual;
        this.predicted = predicted;
        this.numClasses = classNames.length;
        this.classNames = classNames.clone();
        this.matrix = new int[numClasses][numClasses];
        
        buildMatrix();
    }
    
    private void buildMatrix() {
        // Initialize matrix
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                matrix[i][j] = 0;
            }
        }
        
        // Populate confusion matrix
        for (int i = 0; i < actual.length; i++) {
            int actualClass = actual[i];
            int predClass = predicted[i];
            
            // Ensure bounds
            if (actualClass >= 0 && actualClass < numClasses && 
                predClass >= 0 && predClass < numClasses) {
                matrix[actualClass][predClass]++;
            }
        }
    }
    
    private String[] generateDefaultClassNames() {
        String[] names = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            names[i] = "Class " + i;
        }
        return names;
    }
    
    @Override
    public void display() {
        System.out.println("📊 " + title);
        System.out.println("=" + "=".repeat(title.length() + 3));
        displayMatrix();
        displayClassificationReport();
        displayStatistics();
    }
    
    /**
     * Enhanced matrix display with Unicode box drawing and color indicators
     */
    public void displayMatrix() {
        System.out.println("\n📊 Enhanced Confusion Matrix (Actual vs Predicted):");
        System.out.println("═".repeat(65));
        
        // Calculate dynamic widths
        int maxWidth = Math.max(12, getMaxClassNameLength() + 2);
        int cellWidth = Math.max(8, String.valueOf(getMaxMatrixValue()).length() + 4);
        
        // Enhanced header with Unicode box drawing
        System.out.print("┌" + "─".repeat(maxWidth) + "┬");
        for (int i = 0; i < numClasses; i++) {
            System.out.print("─".repeat(cellWidth) + (i < numClasses - 1 ? "┬" : "┐"));
        }
        System.out.println();
        
        // Column headers with better formatting
        System.out.printf("│%s%s%s│", " ".repeat((maxWidth-11)/2), "Predicted →", " ".repeat((maxWidth-11)/2));
        for (int i = 0; i < numClasses; i++) {
            String className = classNames[i];
            if (className.length() > cellWidth - 2) {
                className = className.substring(0, cellWidth - 3) + "…";
            }
            int padding = (cellWidth - className.length()) / 2;
            System.out.printf("%s%s%s│", 
                " ".repeat(padding), 
                className, 
                " ".repeat(cellWidth - className.length() - padding));
        }
        System.out.println();
        
        // Separator with enhanced style
        System.out.print("├" + "─".repeat(maxWidth) + "┼");
        for (int i = 0; i < numClasses; i++) {
            System.out.print("─".repeat(cellWidth) + (i < numClasses - 1 ? "┼" : "┤"));
        }
        System.out.println();
        
        // Matrix rows with enhanced formatting and indicators
        for (int i = 0; i < numClasses; i++) {
            String rowLabel = "Act: " + classNames[i];
            if (rowLabel.length() > maxWidth - 2) {
                rowLabel = rowLabel.substring(0, maxWidth - 3) + "…";
            }
            System.out.printf("│%s%s%s│", 
                " ".repeat((maxWidth - rowLabel.length())/2),
                rowLabel,
                " ".repeat(maxWidth - rowLabel.length() - (maxWidth - rowLabel.length())/2));
            
            for (int j = 0; j < numClasses; j++) {
                String cellValue = formatEnhancedCellValue(matrix[i][j], i == j, i, j);
                int padding = (cellWidth - cellValue.length()) / 2;
                System.out.printf("%s%s%s│", 
                    " ".repeat(padding), 
                    cellValue, 
                    " ".repeat(cellWidth - cellValue.length() - padding));
            }
            System.out.println();
        }
        
        // Bottom border
        System.out.print("└" + "─".repeat(maxWidth) + "┴");
        for (int i = 0; i < numClasses; i++) {
            System.out.print("─".repeat(cellWidth) + (i < numClasses - 1 ? "┴" : "┘"));
        }
        System.out.println();
        
        // Enhanced legend and performance indicators
        displayEnhancedLegend();
    }
    
    private String formatEnhancedCellValue(int value, boolean isDiagonal, int actualClass, int predClass) {
        if (value == 0) {
            return "  ·  ";
        }
        
        if (isDiagonal) {
            // True positives - use different symbols based on value magnitude
            if (value >= 50) return "🟢 " + value;
            else if (value >= 20) return "✅ " + value;
            else return "☑️ " + value;
        } else {
            // False positives/negatives - indicate severity
            double errorRate = (double) value / getRowSum(actualClass);
            if (errorRate >= 0.3) return "🔴 " + value; // High error
            else if (errorRate >= 0.1) return "🟡 " + value; // Medium error
            else return "🟠 " + value; // Low error
        }
    }
    
    private void displayEnhancedLegend() {
        System.out.println("\n🔍 Enhanced Legend & Performance Indicators:");
        System.out.println("═".repeat(50));
        System.out.println("  🟢 = Excellent predictions (≥50 samples)");
        System.out.println("  ✅ = Good predictions (20-49 samples)");
        System.out.println("  ☑️ = Fair predictions (1-19 samples)");
        System.out.println("  🔴 = High error rate (≥30% misclassification)");
        System.out.println("  🟡 = Medium error rate (10-29% misclassification)");
        System.out.println("  🟠 = Low error rate (<10% misclassification)");
        System.out.println("  ·  = No predictions");
        System.out.println("\n📈 Matrix Quality Assessment:");
        assessMatrixQuality();
    }
    
    private void assessMatrixQuality() {
        double accuracy = calculateAccuracy();
        int totalPredictions = actual.length;
        int strongDiagonal = 0;
        int weakOffDiagonal = 0;
        
        for (int i = 0; i < numClasses; i++) {
            if (matrix[i][i] >= totalPredictions * 0.1) strongDiagonal++;
            for (int j = 0; j < numClasses; j++) {
                if (i != j && matrix[i][j] <= totalPredictions * 0.05) weakOffDiagonal++;
            }
        }
        
        // Overall assessment
        String quality;
        if (accuracy >= 0.9 && strongDiagonal == numClasses) {
            quality = "🟢 Excellent - Strong diagonal, minimal confusion";
        } else if (accuracy >= 0.7 && strongDiagonal >= numClasses * 0.7) {
            quality = "🟡 Good - Clear class separation";
        } else if (accuracy >= 0.5) {
            quality = "🟠 Fair - Some class confusion present";
        } else {
            quality = "🔴 Poor - Significant class confusion";
        }
        
        System.out.println("  Overall Quality: " + quality);
        System.out.printf("  Accuracy: %.1f%%, Strong Classes: %d/%d\n", 
            accuracy * 100, strongDiagonal, numClasses);
    }
    
    private int getMaxMatrixValue() {
        int max = 0;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                max = Math.max(max, matrix[i][j]);
            }
        }
        return max;
    }
    
    private int getRowSum(int row) {
        int sum = 0;
        for (int j = 0; j < numClasses; j++) {
            sum += matrix[row][j];
        }
        return sum;
    }
    
    /**
     * Display detailed classification report
     */
    public void displayClassificationReport() {
        System.out.println("📈 Classification Report:");
        System.out.println("========================");
        
        // Header
        System.out.printf("%-12s %10s %10s %10s %10s\n", 
            "Class", "Precision", "Recall", "F1-Score", "Support");
        System.out.println("-".repeat(62));
        
        double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
        int totalSupport = 0;
        int validClasses = 0;
        
        // Per-class metrics
        for (int i = 0; i < numClasses; i++) {
            double precision = calculatePrecision(i);
            double recall = calculateRecall(i);
            double f1 = calculateF1Score(i);
            int support = getClassSupport(i);
            
            if (!Double.isNaN(precision)) {
                totalPrecision += precision;
                totalRecall += recall;
                totalF1 += f1;
                validClasses++;
            }
            totalSupport += support;
            
            System.out.printf("%-12s %10.3f %10.3f %10.3f %10d\n", 
                classNames[i], 
                Double.isNaN(precision) ? 0.0 : precision,
                Double.isNaN(recall) ? 0.0 : recall,
                Double.isNaN(f1) ? 0.0 : f1,
                support);
        }
        
        System.out.println("-".repeat(62));
        
        // Averages
        if (validClasses > 0) {
            System.out.printf("%-12s %10.3f %10.3f %10.3f %10d\n", 
                "Macro Avg", 
                totalPrecision / validClasses,
                totalRecall / validClasses, 
                totalF1 / validClasses,
                totalSupport);
                
            double accuracy = calculateAccuracy();
            System.out.printf("%-12s %10.3f %10.3f %10.3f %10d\n", 
                "Accuracy", accuracy, accuracy, accuracy, totalSupport);
        }
        System.out.println();
    }
    
    /**
     * Display additional statistics
     */
    public void displayStatistics() {
        System.out.println("🔢 Matrix Statistics:");
        System.out.println("====================");
        System.out.println("Total samples: " + actual.length);
        System.out.println("Number of classes: " + numClasses);
        System.out.println("Accuracy: " + String.format("%.3f", calculateAccuracy()));
        System.out.println("True positives (diagonal): " + getTotalTruePositives());
        System.out.println("False predictions: " + (actual.length - getTotalTruePositives()));
        
        // Balanced accuracy
        double balancedAccuracy = calculateBalancedAccuracy();
        if (!Double.isNaN(balancedAccuracy)) {
            System.out.println("Balanced accuracy: " + String.format("%.3f", balancedAccuracy));
        }
        System.out.println();
    }
    
    // Calculation methods
    
    /**
     * Calculate precision for a specific class
     */
    public double calculatePrecision(int classIndex) {
        int truePositives = matrix[classIndex][classIndex];
        int falsePositives = 0;
        
        // Sum column (all predictions for this class)
        for (int i = 0; i < numClasses; i++) {
            if (i != classIndex) {
                falsePositives += matrix[i][classIndex];
            }
        }
        
        int totalPredicted = truePositives + falsePositives;
        return totalPredicted == 0 ? Double.NaN : (double) truePositives / totalPredicted;
    }
    
    /**
     * Calculate recall for a specific class
     */
    public double calculateRecall(int classIndex) {
        int truePositives = matrix[classIndex][classIndex];
        int falseNegatives = 0;
        
        // Sum row (all actual instances of this class)
        for (int j = 0; j < numClasses; j++) {
            if (j != classIndex) {
                falseNegatives += matrix[classIndex][j];
            }
        }
        
        int totalActual = truePositives + falseNegatives;
        return totalActual == 0 ? Double.NaN : (double) truePositives / totalActual;
    }
    
    /**
     * Calculate F1-score for a specific class
     */
    public double calculateF1Score(int classIndex) {
        double precision = calculatePrecision(classIndex);
        double recall = calculateRecall(classIndex);
        
        if (Double.isNaN(precision) || Double.isNaN(recall) || (precision + recall) == 0) {
            return Double.NaN;
        }
        
        return 2 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Calculate overall accuracy
     */
    public double calculateAccuracy() {
        return (double) getTotalTruePositives() / actual.length;
    }
    
    /**
     * Calculate balanced accuracy (average of per-class recalls)
     */
    public double calculateBalancedAccuracy() {
        double totalRecall = 0;
        int validClasses = 0;
        
        for (int i = 0; i < numClasses; i++) {
            double recall = calculateRecall(i);
            if (!Double.isNaN(recall)) {
                totalRecall += recall;
                validClasses++;
            }
        }
        
        return validClasses == 0 ? Double.NaN : totalRecall / validClasses;
    }
    
    /**
     * Get support (number of actual instances) for a class
     */
    public int getClassSupport(int classIndex) {
        int support = 0;
        for (int j = 0; j < numClasses; j++) {
            support += matrix[classIndex][j];
        }
        return support;
    }
    
    private int getTotalTruePositives() {
        int total = 0;
        for (int i = 0; i < numClasses; i++) {
            total += matrix[i][i];
        }
        return total;
    }
    
    private int getMaxClassNameLength() {
        int max = 0;
        for (String name : classNames) {
            max = Math.max(max, name.length());
        }
        return max;
    }
    
    // Getters and setters
    
    public int[][] getMatrix() {
        return matrix;
    }
    
    public int getNumClasses() {
        return numClasses;
    }
    
    public String[] getClassNames() {
        return classNames.clone();
    }
    
    public void setClassNames(String[] classNames) {
        if (classNames.length != numClasses) {
            throw new IllegalArgumentException("Number of class names must match number of classes");
        }
        this.classNames = classNames.clone();
    }
    
    @Override
    public void setTitle(String title) {
        this.title = title;
    }
    
    @Override
    public String getTitle() {
        return title;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Confusion Matrix:\n");
        sb.append("Accuracy: ").append(String.format("%.3f", calculateAccuracy())).append("\n");
        sb.append("Classes: ").append(numClasses).append("\n");
        sb.append("Samples: ").append(actual.length).append("\n");
        return sb.toString();
    }
}
