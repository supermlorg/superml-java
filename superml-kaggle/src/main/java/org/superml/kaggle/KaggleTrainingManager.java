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

package org.superml.kaggle;

import org.superml.core.Estimator;
import org.superml.kaggle.api.KaggleClient;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * High-level manager for Kaggle competition workflows.
 * Provides end-to-end automation for competition participation.
 */
public class KaggleTrainingManager {
    
    private final KaggleClient kaggleClient;
    private final String workingDirectory;
    
    public KaggleTrainingManager(KaggleClient kaggleClient, String workingDirectory) {
        this.kaggleClient = kaggleClient;
        this.workingDirectory = workingDirectory;
    }
    
    /**
     * Create a manager with automatic Kaggle client initialization.
     */
    public static KaggleTrainingManager create(String workingDirectory) throws IOException {
        KaggleClient client = KaggleClient.fromCredentialsFile();
        return new KaggleTrainingManager(client, workingDirectory);
    }
    
    /**
     * Download competition data and return file paths.
     * @param competitionName Name of the Kaggle competition
     * @return List of downloaded file paths
     */
    public List<String> downloadCompetitionData(String competitionName) throws IOException {
        System.out.println("📥 Downloading competition data for: " + competitionName);
        
        String dataPath = workingDirectory + "/" + competitionName;
        List<String> dataFiles = kaggleClient.downloadCompetitionData(competitionName, dataPath);
        
        System.out.println("-> Downloaded " + dataFiles.size() + " files");
        return dataFiles;
    }
    
    /**
     * Get competition leaderboard and display information.
     */
    public void showLeaderboard(String competitionName) throws IOException {
        KaggleClient.LeaderboardInfo leaderboard = kaggleClient.getLeaderboard(competitionName);
        
        System.out.println("🏆 Competition Leaderboard: " + competitionName);
        System.out.println("=" .repeat(60));
        System.out.printf("%-30s %-15s %-20s\n", "Team", "Score", "Date");
        System.out.println("-".repeat(60));
        
        int rank = 1;
        for (KaggleClient.LeaderboardEntry entry : leaderboard.entries) {
            System.out.printf("%3d. %-26s %-15.6f %-20s\n", 
                rank++, entry.teamName, entry.score, entry.submissionDate);
            
            if (rank > 10) { // Show top 10
                System.out.println("    ... (showing top 10 entries)");
                break;
            }
        }
    }
    
    /**
     * Submit predictions to a competition.
     */
    public KaggleClient.SubmissionResult submitPredictions(String competitionName, 
                                                          double[] predictions, 
                                                          String[] testIds, 
                                                          String submissionMessage) throws IOException {
        System.out.println("📝 Creating submission file...");
        String submissionPath = createSubmissionFile(competitionName, testIds, predictions);
        
        System.out.println("🚀 Submitting to Kaggle...");
        return kaggleClient.submitPredictions(competitionName, submissionPath, submissionMessage);
    }
    
    private String createSubmissionFile(String competitionName, String[] testIds, double[] predictions) throws IOException {
        String submissionPath = workingDirectory + "/" + competitionName + "_submission.csv";
        
        try (PrintWriter writer = new PrintWriter(submissionPath)) {
            writer.println("Id,Target");
            for (int i = 0; i < testIds.length; i++) {
                writer.printf("%s,%.6f\n", testIds[i], predictions[i]);
            }
        }
        
        return submissionPath;
    }
}
