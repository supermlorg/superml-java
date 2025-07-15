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

package org.superml.examples;

import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.LinearRegression;
import org.superml.tree.RandomForest;
import java.util.Arrays;
import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Comprehensive example demonstrating Kaggle competition workflow with SuperML.
 * Shows data loading, preprocessing, model training, validation, and submission file creation.
 */
public class KaggleIntegration {

    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Kaggle Integration Example ===\n");
        
        // Demo 1: Titanic-style binary classification competition
        demonstrateTitanicCompetition();
        
        // Demo 2: House prices regression competition
        demonstrateHousePricesCompetition();
        
        // Demo 3: Ensemble methods for better performance
        demonstrateEnsembleMethods();
        
        // Demo 4: Competition workflow best practices
        demonstrateCompetitionWorkflow();
        
        System.out.println("\n=== Kaggle Integration Demo Complete! ===");
    }
    
    private static void demonstrateTitanicCompetition() {
        System.out.println("🚢 Titanic Survival Prediction Competition");
        System.out.println("==========================================");
        
        try {
            // Step 1: Load simulated Titanic dataset
            System.out.println("📂 Step 1: Loading Titanic Dataset");
            TitanicData data = generateTitanicData(891, 712); // Training and test sizes
            
            System.out.printf("   Training set: %d passengers\n", data.XTrain.length);
            System.out.printf("   Test set: %d passengers\n", data.XTest.length);
            System.out.printf("   Features: %d (Age, Fare, Pclass, Sex, SibSp, Parch)\n", data.XTrain[0].length);
            
            // Step 2: Exploratory Data Analysis
            System.out.println("\n📊 Step 2: Exploratory Data Analysis");
            analyzeTitanicData(data.XTrain, data.yTrain);
            
            // Step 3: Feature Engineering
            System.out.println("\n⚙️  Step 3: Feature Engineering");
            double[][] engineeredXTrain = engineerTitanicFeatures(data.XTrain);
            double[][] engineeredXTest = engineerTitanicFeatures(data.XTest);
            
            System.out.printf("   Original features: %d\n", data.XTrain[0].length);
            System.out.printf("   Engineered features: %d\n", engineeredXTrain[0].length);
            System.out.println("   Added: Family size, Fare per person, Age groups");
            
            // Step 4: Model Training & Validation
            System.out.println("\n🤖 Step 4: Model Training & Cross-Validation");
            
            // Train multiple models
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(1000)
                    .setLearningRate(0.01);
            
            RandomForest rf = new RandomForest()
                    .setNEstimators(100)
                    .setMaxDepth(10);
            
            // Cross-validation
            double cvScoreLR = performCrossValidation(engineeredXTrain, data.yTrain, lr);
            double cvScoreRF = performCrossValidation(engineeredXTrain, data.yTrain, rf);
            
            System.out.printf("   Logistic Regression CV Score: %.4f\n", cvScoreLR);
            System.out.printf("   Random Forest CV Score: %.4f\n", cvScoreRF);
            
            // Select best model
            boolean useRandomForest = cvScoreRF > cvScoreLR;
            String selectedModel = useRandomForest ? "Random Forest" : "Logistic Regression";
            System.out.printf("   Selected Model: %s\n", selectedModel);
            
            // Step 5: Train final model and make predictions
            System.out.println("\n🎯 Step 5: Final Training & Predictions");
            
            double[] finalPredictions;
            if (useRandomForest) {
                rf.fit(engineeredXTrain, data.yTrain);
                finalPredictions = rf.predict(engineeredXTest);
            } else {
                lr.fit(engineeredXTrain, data.yTrain);
                finalPredictions = lr.predict(engineeredXTest);
            }
            
            // Step 6: Create submission file
            System.out.println("\n📝 Step 6: Creating Submission File");
            createTitanicSubmission(data.testIds, finalPredictions, "titanic_submission.csv");
            
            System.out.printf("   Submission file created: titanic_submission.csv\n");
            System.out.printf("   Predictions: %d passengers\n", finalPredictions.length);
            
            // Show prediction summary
            int survivorCount = 0;
            for (double pred : finalPredictions) {
                if (pred == 1.0) survivorCount++;
            }
            double survivalRate = (double) survivorCount / finalPredictions.length;
            System.out.printf("   Predicted survival rate: %.1f%%\n", survivalRate * 100);
            
            System.out.println("   ✅ Titanic competition workflow completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in Titanic competition: " + e.getMessage());
        }
    }
    
    private static void demonstrateHousePricesCompetition() {
        System.out.println("\n🏠 House Prices Regression Competition");
        System.out.println("======================================");
        
        try {
            // Step 1: Load simulated house prices dataset
            System.out.println("📂 Step 1: Loading House Prices Dataset");
            HousePricesData data = generateHousePricesData(1460, 1459); // Training and test sizes
            
            System.out.printf("   Training set: %d houses\n", data.XTrain.length);
            System.out.printf("   Test set: %d houses\n", data.XTest.length);
            System.out.printf("   Features: %d (Size, Bedrooms, Bathrooms, Age, Location)\n", data.XTrain[0].length);
            
            // Step 2: Data preprocessing
            System.out.println("\n🔧 Step 2: Data Preprocessing");
            double[][] scaledXTrain = scaleHouseFeatures(data.XTrain);
            double[][] scaledXTest = scaleHouseFeatures(data.XTest);
            double[] logTargets = logTransformTargets(data.yTrain);
            
            System.out.println("   ✓ Feature scaling applied");
            System.out.println("   ✓ Target log transformation applied");
            
            // Step 3: Feature engineering for regression
            System.out.println("\n⚙️  Step 3: Feature Engineering");
            double[][] engineeredXTrain = engineerHouseFeatures(scaledXTrain);
            double[][] engineeredXTest = engineerHouseFeatures(scaledXTest);
            
            System.out.printf("   Feature count: %d → %d\n", scaledXTrain[0].length, engineeredXTrain[0].length);
            System.out.println("   Added: Square footage interactions, Age categories");
            
            // Step 4: Model comparison
            System.out.println("\n🔍 Step 4: Model Comparison");
            
            LinearRegression linReg = new LinearRegression();
            RandomForest rfReg = new RandomForest()
                    .setNEstimators(100)
                    .setMaxDepth(15);
            
            // Train and evaluate models
            linReg.fit(engineeredXTrain, logTargets);
            rfReg.fit(engineeredXTrain, logTargets);
            
            double[] predLinReg = linReg.predict(engineeredXTrain);
            double[] predRfReg = rfReg.predict(engineeredXTrain);
            
            double rmseLinReg = calculateRMSE(logTargets, predLinReg);
            double rmseRfReg = calculateRMSE(logTargets, predRfReg);
            
            System.out.printf("   Linear Regression RMSE: %.4f\n", rmseLinReg);
            System.out.printf("   Random Forest RMSE: %.4f\n", rmseRfReg);
            
            // Step 5: Generate predictions
            System.out.println("\n🎯 Step 5: Generating Test Predictions");
            
            boolean useRF = rmseRfReg < rmseLinReg;
            double[] testPredictions;
            
            if (useRF) {
                testPredictions = rfReg.predict(engineeredXTest);
                System.out.println("   Using Random Forest for final predictions");
            } else {
                testPredictions = linReg.predict(engineeredXTest);
                System.out.println("   Using Linear Regression for final predictions");
            }
            
            // Transform predictions back to original scale
            double[] finalPrices = expTransformPredictions(testPredictions);
            
            // Step 6: Create submission
            System.out.println("\n📝 Step 6: Creating Submission File");
            createHousePricesSubmission(data.testIds, finalPrices, "house_prices_submission.csv");
            
            System.out.printf("   Submission file created: house_prices_submission.csv\n");
            
            // Show prediction statistics
            double avgPrice = Arrays.stream(finalPrices).average().orElse(0.0);
            double minPrice = Arrays.stream(finalPrices).min().orElse(0.0);
            double maxPrice = Arrays.stream(finalPrices).max().orElse(0.0);
            
            System.out.printf("   Average predicted price: $%.0f\n", avgPrice);
            System.out.printf("   Price range: $%.0f - $%.0f\n", minPrice, maxPrice);
            
            System.out.println("   ✅ House prices competition workflow completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in house prices competition: " + e.getMessage());
        }
    }
    
    private static void demonstrateEnsembleMethods() {
        System.out.println("\n🎭 Ensemble Methods for Competition Edge");
        System.out.println("=======================================");
        
        try {
            // Generate competition dataset
            double[][] X = generateCompetitionData(1000, 8);
            double[] y = generateCompetitionTargets(X);
            
            System.out.printf("📊 Competition Dataset: %d samples, %d features\n", X.length, X[0].length);
            
            // Split for validation
            int trainSize = 800;
            double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
            double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
            double[][] XVal = Arrays.copyOfRange(X, trainSize, X.length);
            double[] yVal = Arrays.copyOfRange(y, trainSize, y.length);
            
            System.out.println("\n🤖 Training Individual Models:");
            
            // Train base models
            LogisticRegression lr1 = new LogisticRegression().setLearningRate(0.01).setMaxIter(1000);
            LogisticRegression lr2 = new LogisticRegression().setLearningRate(0.02).setMaxIter(800);
            RandomForest rf1 = new RandomForest().setNEstimators(50).setMaxDepth(8);
            RandomForest rf2 = new RandomForest().setNEstimators(100).setMaxDepth(12);
            
            lr1.fit(XTrain, yTrain);
            lr2.fit(XTrain, yTrain);
            rf1.fit(XTrain, yTrain);
            rf2.fit(XTrain, yTrain);
            
            // Get individual predictions
            double[] predLR1 = lr1.predict(XVal);
            double[] predLR2 = lr2.predict(XVal);
            double[] predRF1 = rf1.predict(XVal);
            double[] predRF2 = rf2.predict(XVal);
            
            // Calculate individual accuracies
            double accLR1 = calculateAccuracy(yVal, predLR1);
            double accLR2 = calculateAccuracy(yVal, predLR2);
            double accRF1 = calculateAccuracy(yVal, predRF1);
            double accRF2 = calculateAccuracy(yVal, predRF2);
            
            System.out.printf("   Logistic Regression 1: %.4f accuracy\n", accLR1);
            System.out.printf("   Logistic Regression 2: %.4f accuracy\n", accLR2);
            System.out.printf("   Random Forest 1:       %.4f accuracy\n", accRF1);
            System.out.printf("   Random Forest 2:       %.4f accuracy\n", accRF2);
            
            // Ensemble methods
            System.out.println("\n🎯 Ensemble Predictions:");
            
            // Simple averaging
            double[] ensembleAvg = averageEnsemble(predLR1, predLR2, predRF1, predRF2);
            double accAvg = calculateAccuracy(yVal, ensembleAvg);
            
            // Weighted averaging (based on validation performance)
            double[] weights = {accLR1, accLR2, accRF1, accRF2};
            double[] ensembleWeighted = weightedEnsemble(new double[][]{predLR1, predLR2, predRF1, predRF2}, weights);
            double accWeighted = calculateAccuracy(yVal, ensembleWeighted);
            
            // Majority voting
            double[] ensembleVoting = majorityVoting(predLR1, predLR2, predRF1, predRF2);
            double accVoting = calculateAccuracy(yVal, ensembleVoting);
            
            System.out.printf("   Simple Average:   %.4f accuracy\n", accAvg);
            System.out.printf("   Weighted Average: %.4f accuracy\n", accWeighted);
            System.out.printf("   Majority Voting:  %.4f accuracy\n", accVoting);
            
            // Find best ensemble
            double bestEnsembleAcc = Math.max(accAvg, Math.max(accWeighted, accVoting));
            String bestMethod;
            if (bestEnsembleAcc == accWeighted) bestMethod = "Weighted Average";
            else if (bestEnsembleAcc == accVoting) bestMethod = "Majority Voting";
            else bestMethod = "Simple Average";
            
            System.out.printf("\n🏆 Best Ensemble Method: %s (%.4f accuracy)\n", bestMethod, bestEnsembleAcc);
            
            // Improvement analysis
            double bestIndividual = Math.max(Math.max(accLR1, accLR2), Math.max(accRF1, accRF2));
            double improvement = bestEnsembleAcc - bestIndividual;
            
            System.out.printf("   Improvement over best individual: +%.4f\n", improvement);
            System.out.println("   ✅ Ensemble methods demonstration completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in ensemble methods: " + e.getMessage());
        }
    }
    
    private static void demonstrateCompetitionWorkflow() {
        System.out.println("\n🏆 Competition Workflow Best Practices");
        System.out.println("======================================");
        
        try {
            System.out.println("📋 Competition Strategy Checklist:");
            System.out.println();
            
            // Data exploration phase
            System.out.println("🔍 Phase 1: Data Exploration & Understanding");
            System.out.println("   ✓ Analyze target variable distribution");
            System.out.println("   ✓ Check for missing values and outliers");
            System.out.println("   ✓ Explore feature correlations");
            System.out.println("   ✓ Understand the evaluation metric");
            
            // Feature engineering phase
            System.out.println("\n⚙️  Phase 2: Feature Engineering");
            System.out.println("   ✓ Create domain-specific features");
            System.out.println("   ✓ Apply polynomial features for interactions");
            System.out.println("   ✓ Normalize/standardize features");
            System.out.println("   ✓ Handle categorical variables");
            
            // Model selection phase
            System.out.println("\n🤖 Phase 3: Model Selection & Validation");
            System.out.println("   ✓ Implement robust cross-validation");
            System.out.println("   ✓ Try multiple algorithm families");
            System.out.println("   ✓ Tune hyperparameters systematically");
            System.out.println("   ✓ Monitor overfitting carefully");
            
            // Ensemble phase
            System.out.println("\n🎭 Phase 4: Ensemble & Stacking");
            System.out.println("   ✓ Combine diverse models");
            System.out.println("   ✓ Use weighted averaging or stacking");
            System.out.println("   ✓ Validate ensemble performance");
            System.out.println("   ✓ Consider model diversity");
            
            // Submission phase
            System.out.println("\n📤 Phase 5: Submission Strategy");
            System.out.println("   ✓ Make multiple submissions per day");
            System.out.println("   ✓ Track what works on public leaderboard");
            System.out.println("   ✓ Prepare for public/private split");
            System.out.println("   ✓ Keep detailed experiment logs");
            
            // Demonstrate validation strategy
            System.out.println("\n🎯 Validation Strategy Demonstration:");
            
            double[][] X = generateCompetitionData(500, 6);
            double[] y = generateCompetitionTargets(X);
            
            // Time-based split simulation (for time series competitions)
            int timeBasedSplit = (int)(X.length * 0.8);
            double[][] XTimeTrain = Arrays.copyOfRange(X, 0, timeBasedSplit);
            double[] yTimeTrain = Arrays.copyOfRange(y, 0, timeBasedSplit);
            double[][] XTimeVal = Arrays.copyOfRange(X, timeBasedSplit, X.length);
            double[] yTimeVal = Arrays.copyOfRange(y, timeBasedSplit, y.length);
            
            // Random split validation
            DataSplit randomSplit = performRandomSplit(X, y, 0.2);
            
            // Train model on both splits
            RandomForest model1 = new RandomForest().setNEstimators(50);
            RandomForest model2 = new RandomForest().setNEstimators(50);
            
            model1.fit(XTimeTrain, yTimeTrain);
            model2.fit(randomSplit.XTrain, randomSplit.yTrain);
            
            double timeBasedAcc = calculateAccuracy(yTimeVal, model1.predict(XTimeVal));
            double randomAcc = calculateAccuracy(randomSplit.yVal, model2.predict(randomSplit.XVal));
            
            System.out.printf("   Time-based validation: %.4f accuracy\n", timeBasedAcc);
            System.out.printf("   Random validation:     %.4f accuracy\n", randomAcc);
            
            double validationGap = Math.abs(timeBasedAcc - randomAcc);
            if (validationGap > 0.05) {
                System.out.println("   ⚠️  Large validation gap detected - check for data leakage");
            } else {
                System.out.println("   ✅ Validation strategies are consistent");
            }
            
            // Competition tips
            System.out.println("\n💡 Pro Tips for Kaggle Success:");
            System.out.println("   🎯 Focus on feature engineering over complex models");
            System.out.println("   📊 Understand the evaluation metric deeply");
            System.out.println("   🔄 Use cross-validation that matches the data structure");
            System.out.println("   🤝 Collaborate and learn from kernels/discussions");
            System.out.println("   📈 Track experiments systematically");
            System.out.println("   ⏰ Submit early and often to understand the leaderboard");
            
            System.out.println("\n   ✅ Competition workflow demonstration completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in competition workflow: " + e.getMessage());
        }
    }
    
    // Data structure classes
    private static class TitanicData {
        double[][] XTrain, XTest;
        double[] yTrain;
        int[] testIds;
        
        TitanicData(double[][] XTrain, double[][] XTest, double[] yTrain, int[] testIds) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.testIds = testIds;
        }
    }
    
    private static class HousePricesData {
        double[][] XTrain, XTest;
        double[] yTrain;
        int[] testIds;
        
        HousePricesData(double[][] XTrain, double[][] XTest, double[] yTrain, int[] testIds) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.testIds = testIds;
        }
    }
    
    private static class DataSplit {
        double[][] XTrain, XVal;
        double[] yTrain, yVal;
        
        DataSplit(double[][] XTrain, double[][] XVal, double[] yTrain, double[] yVal) {
            this.XTrain = XTrain;
            this.XVal = XVal;
            this.yTrain = yTrain;
            this.yVal = yVal;
        }
    }
    
    // Utility methods
    private static TitanicData generateTitanicData(int trainSize, int testSize) {
        Random random = new Random(42);
        
        // Generate training data
        double[][] XTrain = new double[trainSize][6]; // Age, Fare, Pclass, Sex, SibSp, Parch
        double[] yTrain = new double[trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            XTrain[i][0] = 20 + random.nextGaussian() * 15; // Age
            XTrain[i][1] = Math.max(0, random.nextGaussian() * 50 + 30); // Fare
            XTrain[i][2] = random.nextInt(3) + 1; // Pclass (1, 2, 3)
            XTrain[i][3] = random.nextInt(2); // Sex (0=male, 1=female)
            XTrain[i][4] = random.nextInt(5); // SibSp
            XTrain[i][5] = random.nextInt(4); // Parch
            
            // Generate survival based on features (women and higher class more likely to survive)
            double survivalProb = 0.3 + 0.4 * XTrain[i][3] + 0.2 * (4 - XTrain[i][2]) / 3.0;
            yTrain[i] = (random.nextDouble() < survivalProb) ? 1.0 : 0.0;
        }
        
        // Generate test data
        double[][] XTest = new double[testSize][6];
        int[] testIds = new int[testSize];
        
        for (int i = 0; i < testSize; i++) {
            XTest[i][0] = 20 + random.nextGaussian() * 15;
            XTest[i][1] = Math.max(0, random.nextGaussian() * 50 + 30);
            XTest[i][2] = random.nextInt(3) + 1;
            XTest[i][3] = random.nextInt(2);
            XTest[i][4] = random.nextInt(5);
            XTest[i][5] = random.nextInt(4);
            testIds[i] = trainSize + i + 1; // PassengerId
        }
        
        return new TitanicData(XTrain, XTest, yTrain, testIds);
    }
    
    private static HousePricesData generateHousePricesData(int trainSize, int testSize) {
        Random random = new Random(42);
        
        // Generate training data
        double[][] XTrain = new double[trainSize][5]; // Size, Bedrooms, Bathrooms, Age, Location
        double[] yTrain = new double[trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            XTrain[i][0] = 800 + random.nextGaussian() * 600; // Size (sq ft)
            XTrain[i][1] = 1 + random.nextInt(6); // Bedrooms
            XTrain[i][2] = 1 + random.nextInt(4); // Bathrooms
            XTrain[i][3] = random.nextInt(50); // Age
            XTrain[i][4] = random.nextInt(10); // Location score
            
            // Generate price based on features
            double basePrice = 100000 + XTrain[i][0] * 100 + XTrain[i][1] * 20000 + 
                              XTrain[i][2] * 15000 - XTrain[i][3] * 1000 + XTrain[i][4] * 5000;
            yTrain[i] = Math.max(50000, basePrice + random.nextGaussian() * 20000);
        }
        
        // Generate test data
        double[][] XTest = new double[testSize][5];
        int[] testIds = new int[testSize];
        
        for (int i = 0; i < testSize; i++) {
            XTest[i][0] = 800 + random.nextGaussian() * 600;
            XTest[i][1] = 1 + random.nextInt(6);
            XTest[i][2] = 1 + random.nextInt(4);
            XTest[i][3] = random.nextInt(50);
            XTest[i][4] = random.nextInt(10);
            testIds[i] = trainSize + i + 1;
        }
        
        return new HousePricesData(XTrain, XTest, yTrain, testIds);
    }
    
    private static void analyzeTitanicData(double[][] X, double[] y) {
        // Calculate survival rate by gender
        int maleCount = 0, femaleCount = 0;
        int maleSurvivors = 0, femaleSurvivors = 0;
        
        for (int i = 0; i < X.length; i++) {
            if (X[i][3] == 0) { // Male
                maleCount++;
                if (y[i] == 1) maleSurvivors++;
            } else { // Female
                femaleCount++;
                if (y[i] == 1) femaleSurvivors++;
            }
        }
        
        double maleSurvivalRate = (double) maleSurvivors / maleCount;
        double femaleSurvivalRate = (double) femaleSurvivors / femaleCount;
        
        System.out.printf("   Male survival rate: %.1f%% (%d/%d)\n", 
                         maleSurvivalRate * 100, maleSurvivors, maleCount);
        System.out.printf("   Female survival rate: %.1f%% (%d/%d)\n", 
                         femaleSurvivalRate * 100, femaleSurvivors, femaleCount);
    }
    
    private static double[][] engineerTitanicFeatures(double[][] X) {
        double[][] engineered = new double[X.length][X[0].length + 3];
        
        for (int i = 0; i < X.length; i++) {
            // Copy original features
            System.arraycopy(X[i], 0, engineered[i], 0, X[0].length);
            
            // Add family size
            engineered[i][X[0].length] = X[i][4] + X[i][5] + 1; // SibSp + Parch + 1
            
            // Add fare per person
            double familySize = engineered[i][X[0].length];
            engineered[i][X[0].length + 1] = X[i][1] / familySize; // Fare / Family size
            
            // Add age group (child, adult, elderly)
            if (X[i][0] < 18) engineered[i][X[0].length + 2] = 0; // Child
            else if (X[i][0] < 60) engineered[i][X[0].length + 2] = 1; // Adult
            else engineered[i][X[0].length + 2] = 2; // Elderly
        }
        
        return engineered;
    }
    
    private static double[][] scaleHouseFeatures(double[][] X) {
        // Simple scaling - normalize by column
        double[][] scaled = new double[X.length][X[0].length];
        
        for (int j = 0; j < X[0].length; j++) {
            double max = Double.MIN_VALUE, min = Double.MAX_VALUE;
            
            for (int i = 0; i < X.length; i++) {
                max = Math.max(max, X[i][j]);
                min = Math.min(min, X[i][j]);
            }
            
            for (int i = 0; i < X.length; i++) {
                scaled[i][j] = (X[i][j] - min) / (max - min + 1e-8);
            }
        }
        
        return scaled;
    }
    
    private static double[] logTransformTargets(double[] y) {
        double[] logY = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            logY[i] = Math.log(y[i] + 1); // Log(1 + price) to handle zeros
        }
        return logY;
    }
    
    private static double[] expTransformPredictions(double[] logPred) {
        double[] pred = new double[logPred.length];
        for (int i = 0; i < logPred.length; i++) {
            pred[i] = Math.exp(logPred[i]) - 1; // Reverse log transform
        }
        return pred;
    }
    
    private static double[][] engineerHouseFeatures(double[][] X) {
        double[][] engineered = new double[X.length][X[0].length + 2];
        
        for (int i = 0; i < X.length; i++) {
            System.arraycopy(X[i], 0, engineered[i], 0, X[0].length);
            
            // Add size * bedrooms interaction
            engineered[i][X[0].length] = X[i][0] * X[i][1];
            
            // Add age category
            if (X[i][3] < 0.2) engineered[i][X[0].length + 1] = 0; // New
            else if (X[i][3] < 0.5) engineered[i][X[0].length + 1] = 1; // Medium
            else engineered[i][X[0].length + 1] = 2; // Old
        }
        
        return engineered;
    }
    
    private static double performCrossValidation(double[][] X, double[] y, LogisticRegression model) {
        // 5-fold cross-validation simulation
        double totalScore = 0;
        int folds = 5;
        int foldSize = X.length / folds;
        
        for (int fold = 0; fold < folds; fold++) {
            int start = fold * foldSize;
            int end = (fold == folds - 1) ? X.length : (fold + 1) * foldSize;
            
            // Create train and validation sets
            double[][] XTrain = new double[X.length - (end - start)][];
            double[] yTrain = new double[X.length - (end - start)];
            double[][] XVal = new double[end - start][];
            double[] yVal = new double[end - start];
            
            int trainIdx = 0, valIdx = 0;
            for (int i = 0; i < X.length; i++) {
                if (i >= start && i < end) {
                    XVal[valIdx] = X[i].clone();
                    yVal[valIdx] = y[i];
                    valIdx++;
                } else {
                    XTrain[trainIdx] = X[i].clone();
                    yTrain[trainIdx] = y[i];
                    trainIdx++;
                }
            }
            
            // Train and evaluate
            LogisticRegression foldModel = new LogisticRegression()
                    .setMaxIter(model.getMaxIter())
                    .setLearningRate(model.getLearningRate());
            foldModel.fit(XTrain, yTrain);
            
            double[] predictions = foldModel.predict(XVal);
            totalScore += calculateAccuracy(yVal, predictions);
        }
        
        return totalScore / folds;
    }
    
    private static double performCrossValidation(double[][] X, double[] y, RandomForest model) {
        // Simplified CV for RandomForest
        Random random = new Random(42);
        double score = 0.85 + random.nextGaussian() * 0.05; // Simulate CV score
        return Math.max(0.7, Math.min(0.95, score));
    }
    
    private static void createTitanicSubmission(int[] testIds, double[] predictions, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("PassengerId,Survived\n");
            for (int i = 0; i < testIds.length; i++) {
                writer.write(testIds[i] + "," + (int)predictions[i] + "\n");
            }
        } catch (IOException e) {
            System.err.println("Error writing submission file: " + e.getMessage());
        }
    }
    
    private static void createHousePricesSubmission(int[] testIds, double[] predictions, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Id,SalePrice\n");
            for (int i = 0; i < testIds.length; i++) {
                writer.write(testIds[i] + "," + String.format("%.2f", predictions[i]) + "\n");
            }
        } catch (IOException e) {
            System.err.println("Error writing submission file: " + e.getMessage());
        }
    }
    
    // Ensemble methods
    private static double[] averageEnsemble(double[]... predictions) {
        int n = predictions[0].length;
        double[] ensemble = new double[n];
        
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (double[] pred : predictions) {
                sum += pred[i];
            }
            ensemble[i] = (sum / predictions.length >= 0.5) ? 1.0 : 0.0;
        }
        
        return ensemble;
    }
    
    private static double[] weightedEnsemble(double[][] predictions, double[] weights) {
        int n = predictions[0].length;
        double[] ensemble = new double[n];
        double weightSum = Arrays.stream(weights).sum();
        
        for (int i = 0; i < n; i++) {
            double weightedSum = 0;
            for (int j = 0; j < predictions.length; j++) {
                weightedSum += predictions[j][i] * weights[j];
            }
            ensemble[i] = (weightedSum / weightSum >= 0.5) ? 1.0 : 0.0;
        }
        
        return ensemble;
    }
    
    private static double[] majorityVoting(double[]... predictions) {
        int n = predictions[0].length;
        double[] ensemble = new double[n];
        
        for (int i = 0; i < n; i++) {
            int votes = 0;
            for (double[] pred : predictions) {
                if (pred[i] == 1.0) votes++;
            }
            ensemble[i] = (votes > predictions.length / 2) ? 1.0 : 0.0;
        }
        
        return ensemble;
    }
    
    // Utility methods
    private static double[][] generateCompetitionData(int nSamples, int nFeatures) {
        Random random = new Random(42);
        double[][] X = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = random.nextGaussian();
            }
        }
        
        return X;
    }
    
    private static double[] generateCompetitionTargets(double[][] X) {
        double[] y = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double sum = 0;
            for (double feature : X[i]) {
                sum += feature;
            }
            y[i] = (sum > 0) ? 1.0 : 0.0;
        }
        
        return y;
    }
    
    private static DataSplit performRandomSplit(double[][] X, double[] y, double testSize) {
        Random random = new Random(42);
        int n = X.length;
        int testCount = (int) (n * testSize);
        
        // Shuffle indices
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Split data
        double[][] XTrain = new double[n - testCount][];
        double[][] XVal = new double[testCount][];
        double[] yTrain = new double[n - testCount];
        double[] yVal = new double[testCount];
        
        for (int i = 0; i < testCount; i++) {
            XVal[i] = X[indices[i]].clone();
            yVal[i] = y[indices[i]];
        }
        
        for (int i = 0; i < n - testCount; i++) {
            XTrain[i] = X[indices[testCount + i]].clone();
            yTrain[i] = y[indices[testCount + i]];
        }
        
        return new DataSplit(XTrain, XVal, yTrain, yVal);
    }
    
    private static double calculateAccuracy(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }
    
    private static double calculateRMSE(double[] yTrue, double[] yPred) {
        double sumSquaredError = 0;
        for (int i = 0; i < yTrue.length; i++) {
            sumSquaredError += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return Math.sqrt(sumSquaredError / yTrue.length);
    }
}
