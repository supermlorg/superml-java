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

package org.superml.kaggle.api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.entity.mime.content.FileBody;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Kaggle API client for downloading datasets, submitting predictions, and managing competitions.
 * Provides a Java interface to Kaggle's public API.
 */
public class KaggleClient {
    
    private static final String KAGGLE_API_BASE = "https://www.kaggle.com/api/v1";
    private final String username;
    private final String apiKey;
    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    /**
     * Create a new Kaggle client with credentials.
     * @param username Kaggle username
     * @param apiKey Kaggle API key
     */
    public KaggleClient(String username, String apiKey) {
        this.username = username;
        this.apiKey = apiKey;
        this.httpClient = HttpClients.createDefault();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Create a Kaggle client using credentials from kaggle.json file.
     * Looks for kaggle.json in standard locations.
     */
    public static KaggleClient fromCredentialsFile() throws IOException {
        Path credentialsPath = findCredentialsFile();
        if (credentialsPath == null) {
            throw new IOException("Kaggle credentials file not found. Expected locations: " +
                "~/.kaggle/kaggle.json or ./kaggle.json");
        }
        
        JsonNode credentials = new ObjectMapper().readTree(credentialsPath.toFile());
        return new KaggleClient(
            credentials.get("username").asText(),
            credentials.get("key").asText()
        );
    }
    
    private static Path findCredentialsFile() {
        String[] paths = {
            System.getProperty("user.home") + "/.kaggle/kaggle.json",
            "./kaggle.json",
            "kaggle.json"
        };
        
        for (String path : paths) {
            Path p = Paths.get(path);
            if (Files.exists(p)) {
                return p;
            }
        }
        return null;
    }
    
    /**
     * List available competitions.
     * @param category Competition category filter (optional)
     * @param sortBy Sort criteria (optional)
     * @return List of competition information
     */
    public List<CompetitionInfo> listCompetitions(String category, String sortBy) throws IOException {
        StringBuilder url = new StringBuilder(KAGGLE_API_BASE + "/competitions/list");
        
        List<String> params = new ArrayList<>();
        if (category != null) params.add("category=" + category);
        if (sortBy != null) params.add("sortBy=" + sortBy);
        
        if (!params.isEmpty()) {
            url.append("?").append(String.join("&", params));
        }
        
        JsonNode response = makeRequest(url.toString());
        List<CompetitionInfo> competitions = new ArrayList<>();
        
        for (JsonNode comp : response) {
            competitions.add(new CompetitionInfo(
                comp.get("ref").asText(),
                comp.get("title").asText(),
                comp.get("description").asText(),
                comp.get("category").asText(),
                comp.has("deadline") ? comp.get("deadline").asText() : null
            ));
        }
        
        return competitions;
    }
    
    /**
     * Download competition data files.
     * @param competition Competition name/ref
     * @param downloadPath Local directory to save files
     * @return List of downloaded file paths
     */
    public List<String> downloadCompetitionData(String competition, String downloadPath) throws IOException {
        String url = KAGGLE_API_BASE + "/competitions/data/download-all/" + competition;
        
        // Create download directory
        Path downloadDir = Paths.get(downloadPath);
        Files.createDirectories(downloadDir);
        
        // Download zip file
        HttpGet request = new HttpGet(url);
        request.setHeader("Authorization", "Basic " + 
            Base64.getEncoder().encodeToString((username + ":" + apiKey).getBytes()));
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            if (response.getStatusLine().getStatusCode() != 200) {
                throw new IOException("Failed to download data: " + response.getStatusLine().getReasonPhrase());
            }
            
            // Save and extract zip file
            Path zipPath = downloadDir.resolve(competition + ".zip");
            try (InputStream inputStream = response.getEntity().getContent();
                 FileOutputStream outputStream = new FileOutputStream(zipPath.toFile())) {
                inputStream.transferTo(outputStream);
            }
            
            // Extract zip contents
            List<String> extractedFiles = extractZipFile(zipPath, downloadDir);
            
            // Clean up zip file
            Files.deleteIfExists(zipPath);
            
            return extractedFiles;
        }
    }
    
    /**
     * Submit predictions to a competition.
     * @param competition Competition name/ref
     * @param submissionFile Path to submission CSV file
     * @param message Submission message
     * @return Submission result information
     */
    public SubmissionResult submitPredictions(String competition, String submissionFile, String message) throws IOException {
        String url = KAGGLE_API_BASE + "/competitions/submissions/submit/" + competition;
        
        HttpPost request = new HttpPost(url);
        request.setHeader("Authorization", "Basic " + 
            Base64.getEncoder().encodeToString((username + ":" + apiKey).getBytes()));
        
        MultipartEntityBuilder builder = MultipartEntityBuilder.create();
        builder.addPart("file", new FileBody(new File(submissionFile)));
        builder.addTextBody("submissionDescription", message != null ? message : "SuperML Java submission");
        
        request.setEntity(builder.build());
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            JsonNode result = objectMapper.readTree(responseBody);
            
            if (response.getStatusLine().getStatusCode() != 200) {
                throw new IOException("Submission failed: " + result.get("message").asText());
            }
            
            return new SubmissionResult(
                result.get("token").asText(),
                result.get("description").asText(),
                result.get("fileName").asText()
            );
        }
    }
    
    /**
     * Get competition leaderboard.
     * @param competition Competition name/ref
     * @return Leaderboard information
     */
    public LeaderboardInfo getLeaderboard(String competition) throws IOException {
        String url = KAGGLE_API_BASE + "/competitions/leaderboard/view/" + competition;
        JsonNode response = makeRequest(url);
        
        List<LeaderboardEntry> entries = new ArrayList<>();
        for (JsonNode entry : response.get("submissions")) {
            entries.add(new LeaderboardEntry(
                entry.get("teamName").asText(),
                entry.get("score").asDouble(),
                entry.get("submissionDate").asText()
            ));
        }
        
        return new LeaderboardInfo(competition, entries);
    }
    
    private JsonNode makeRequest(String url) throws IOException {
        HttpGet request = new HttpGet(url);
        request.setHeader("Authorization", "Basic " + 
            Base64.getEncoder().encodeToString((username + ":" + apiKey).getBytes()));
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            if (response.getStatusLine().getStatusCode() != 200) {
                throw new IOException("Request failed: " + response.getStatusLine().getReasonPhrase() + " - " + responseBody);
            }
            
            return objectMapper.readTree(responseBody);
        }
    }
    
    private List<String> extractZipFile(Path zipPath, Path extractDir) throws IOException {
        List<String> extractedFiles = new ArrayList<>();
        
        try (InputStream fileInputStream = Files.newInputStream(zipPath);
             ZipArchiveInputStream zipInputStream = new ZipArchiveInputStream(fileInputStream)) {
            
            ZipArchiveEntry entry;
            while ((entry = zipInputStream.getNextZipEntry()) != null) {
                if (!entry.isDirectory()) {
                    Path outputPath = extractDir.resolve(entry.getName());
                    Files.createDirectories(outputPath.getParent());
                    
                    try (OutputStream outputStream = Files.newOutputStream(outputPath)) {
                        zipInputStream.transferTo(outputStream);
                    }
                    
                    extractedFiles.add(outputPath.toString());
                }
            }
        }
        
        return extractedFiles;
    }
    
    /**
     * Close the HTTP client resources.
     */
    public void close() throws IOException {
        httpClient.close();
    }
    
    // Data classes for API responses
    public static class CompetitionInfo {
        public final String ref;
        public final String title;
        public final String description;
        public final String category;
        public final String deadline;
        
        public CompetitionInfo(String ref, String title, String description, String category, String deadline) {
            this.ref = ref;
            this.title = title;
            this.description = description;
            this.category = category;
            this.deadline = deadline;
        }
    }
    
    public static class SubmissionResult {
        public final String token;
        public final String description;
        public final String fileName;
        
        public SubmissionResult(String token, String description, String fileName) {
            this.token = token;
            this.description = description;
            this.fileName = fileName;
        }
    }
    
    public static class LeaderboardInfo {
        public final String competition;
        public final List<LeaderboardEntry> entries;
        
        public LeaderboardInfo(String competition, List<LeaderboardEntry> entries) {
            this.competition = competition;
            this.entries = entries;
        }
    }
    
    public static class LeaderboardEntry {
        public final String teamName;
        public final double score;
        public final String submissionDate;
        
        public LeaderboardEntry(String teamName, double score, String submissionDate) {
            this.teamName = teamName;
            this.score = score;
            this.submissionDate = submissionDate;
        }
    }
}
