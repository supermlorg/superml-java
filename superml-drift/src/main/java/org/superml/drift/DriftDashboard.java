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

package org.superml.drift;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Drift monitoring dashboard and reporting system.
 * Provides real-time monitoring, alerts, and comprehensive reporting
 * for both data drift and concept drift detection.
 */
public class DriftDashboard {
    
    private final DriftConfig config;
    private final DataDriftDetector dataDriftDetector;
    private final ConceptDriftDetector conceptDriftDetector;
    private final Queue<DriftEvent> eventHistory;
    private final ObjectMapper jsonMapper;
    private final SimpleDateFormat dateFormat;
    
    // Alert management
    private Date lastAlertTime;
    private final long alertCooldownMs = 60000; // 1 minute cooldown
    private int totalAlerts = 0;
    
    // Dashboard state
    private boolean isMonitoring = false;
    private Date monitoringStartTime;
    private int totalDataChecks = 0;
    private int totalConceptChecks = 0;
    private int dataDriftDetections = 0;
    private int conceptDriftDetections = 0;
    
    public DriftDashboard(DriftConfig config) {
        this.config = config;
        this.dataDriftDetector = new DataDriftDetector(config);
        this.conceptDriftDetector = new ConceptDriftDetector(config);
        this.eventHistory = new ConcurrentLinkedQueue<>();
        this.jsonMapper = new ObjectMapper();
        this.dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        
        // Configure JSON mapper
        jsonMapper.enable(SerializationFeature.INDENT_OUTPUT);
        jsonMapper.setDateFormat(dateFormat);
    }
    
    /**
     * Start monitoring for drift detection.
     */
    public void startMonitoring() {
        isMonitoring = true;
        monitoringStartTime = new Date();
        
        System.out.println("🎯 Drift monitoring started at " + dateFormat.format(monitoringStartTime));
        System.out.println("📊 Configuration: " + config.toString());
        
        logEvent(new DriftEvent("MONITORING_STARTED", "Drift monitoring session started", 
                               new Date(), DriftEvent.Severity.INFO));
    }
    
    /**
     * Stop monitoring and generate final report.
     */
    public MonitoringReport stopMonitoring() {
        isMonitoring = false;
        Date endTime = new Date();
        
        MonitoringReport report = generateReport(endTime);
        
        System.out.println("🏁 Drift monitoring stopped at " + dateFormat.format(endTime));
        System.out.printf("📈 Session Summary: %d data checks, %d concept checks, %d alerts\n", 
                         totalDataChecks, totalConceptChecks, totalAlerts);
        
        logEvent(new DriftEvent("MONITORING_STOPPED", "Drift monitoring session ended", 
                               endTime, DriftEvent.Severity.INFO));
        
        return report;
    }
    
    /**
     * Check for data drift and update dashboard.
     */
    public DataDriftDetector.DataDriftResult checkDataDrift(double[] referenceData, double[] currentData, 
                                                           String featureName) {
        if (!isMonitoring) {
            throw new IllegalStateException("Monitoring is not active. Call startMonitoring() first.");
        }
        
        totalDataChecks++;
        
        DataDriftDetector.DataDriftResult result = dataDriftDetector.detectDrift(referenceData, currentData, featureName);
        
        if (result.isOverallDrift()) {
            dataDriftDetections++;
            String message = String.format("Data drift detected in feature '%s' using %s (score: %.4f)", 
                                          featureName, result.getPrimaryMethod(), result.getOverallDriftScore());
            
            DriftEvent event = new DriftEvent("DATA_DRIFT", message, new Date(), DriftEvent.Severity.WARNING);
            event.setFeatureName(featureName);
            event.setDriftScore(result.getOverallDriftScore());
            event.setDetectionMethod(result.getPrimaryMethod());
            
            logEvent(event);
            triggerAlert(event);
        }
        
        return result;
    }
    
    /**
     * Check for concept drift and update dashboard.
     */
    public ConceptDriftDetector.ConceptDriftResult checkConceptDrift(double prediction, Double actualLabel, 
                                                                   double confidence) {
        if (!isMonitoring) {
            throw new IllegalStateException("Monitoring is not active. Call startMonitoring() first.");
        }
        
        totalConceptChecks++;
        ConceptDriftDetector.ConceptDriftResult result = conceptDriftDetector.recordPrediction(
            prediction, actualLabel, confidence, new Date());
        
        if (result.isDriftDetected()) {
            conceptDriftDetections++;
            String message = String.format("Concept drift detected using %s (accuracy: %.4f)", 
                                          result.getDetectionMethod(), result.getCurrentAccuracy());
            
            DriftEvent event = new DriftEvent("CONCEPT_DRIFT", message, new Date(), DriftEvent.Severity.CRITICAL);
            event.setDriftScore(1.0 - result.getCurrentAccuracy()); // Higher score = more drift
            event.setDetectionMethod(result.getDetectionMethod());
            
            logEvent(event);
            triggerAlert(event);
        }
        
        return result;
    }
    
    /**
     * Set baseline accuracy for concept drift detection.
     */
    public void setBaselineAccuracy(double accuracy) {
        conceptDriftDetector.setBaselineAccuracy(accuracy);
        
        logEvent(new DriftEvent("BASELINE_SET", 
                               String.format("Baseline accuracy set to %.4f", accuracy), 
                               new Date(), DriftEvent.Severity.INFO));
    }
    
    /**
     * Generate comprehensive monitoring report.
     */
    public MonitoringReport generateReport(Date endTime) {
        long sessionDurationMs = endTime.getTime() - (monitoringStartTime != null ? monitoringStartTime.getTime() : 0);
        
        // Calculate drift rates
        double dataDriftRate = totalDataChecks > 0 ? (double) dataDriftDetections / totalDataChecks : 0.0;
        double conceptDriftRate = totalConceptChecks > 0 ? (double) conceptDriftDetections / totalConceptChecks : 0.0;
        
        // Get current statistics
        ConceptDriftDetector.DriftStatistics conceptStats = conceptDriftDetector.getStatistics();
        
        // Collect recent events
        List<DriftEvent> recentEvents = new ArrayList<>(eventHistory);
        
        return new MonitoringReport(
            monitoringStartTime, endTime, sessionDurationMs,
            totalDataChecks, totalConceptChecks, totalAlerts,
            dataDriftDetections, conceptDriftDetections,
            dataDriftRate, conceptDriftRate,
            conceptStats, recentEvents, config
        );
    }
    
    /**
     * Export monitoring report to JSON file.
     */
    public void exportReportToJson(MonitoringReport report, String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath)) {
            jsonMapper.writeValue(writer, report);
        }
        System.out.println("📄 Report exported to JSON: " + filePath);
    }
    
    /**
     * Export monitoring events to CSV file.
     */
    public void exportEventsToCSV(String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath);
             CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            
            // CSV header
            csvPrinter.printRecord("Timestamp", "Event Type", "Message", "Severity", 
                                  "Feature Name", "Drift Score", "Detection Method");
            
            // Event data
            for (DriftEvent event : eventHistory) {
                csvPrinter.printRecord(
                    dateFormat.format(event.getTimestamp()),
                    event.getEventType(),
                    event.getMessage(),
                    event.getSeverity(),
                    event.getFeatureName() != null ? event.getFeatureName() : "",
                    event.getDriftScore() >= 0 ? String.format("%.4f", event.getDriftScore()) : "",
                    event.getDetectionMethod() != null ? event.getDetectionMethod() : ""
                );
            }
        }
        System.out.println("📊 Events exported to CSV: " + filePath);
    }
    
    /**
     * Get real-time dashboard status.
     */
    public DashboardStatus getDashboardStatus() {
        ConceptDriftDetector.DriftStatistics conceptStats = conceptDriftDetector.getStatistics();
        
        return new DashboardStatus(
            isMonitoring,
            monitoringStartTime,
            totalDataChecks,
            totalConceptChecks,
            dataDriftDetections,
            conceptDriftDetections,
            totalAlerts,
            conceptStats.overallAccuracy,
            conceptStats.slidingWindowAccuracy,
            lastAlertTime,
            eventHistory.size()
        );
    }
    
    /**
     * Print real-time dashboard to console.
     */
    public void printDashboard() {
        DashboardStatus status = getDashboardStatus();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🎯 SUPERML DRIFT MONITORING DASHBOARD");
        System.out.println("=".repeat(60));
        
        System.out.printf("📊 Status: %s\n", status.isMonitoring ? "🟢 MONITORING" : "🔴 STOPPED");
        if (status.monitoringStartTime != null) {
            System.out.printf("⏰ Started: %s\n", dateFormat.format(status.monitoringStartTime));
        }
        
        System.out.println("\n📈 STATISTICS:");
        System.out.printf("  • Data Checks: %d (Drift: %d, Rate: %.2f%%)\n", 
                         status.totalDataChecks, status.dataDriftDetections,
                         status.totalDataChecks > 0 ? 100.0 * status.dataDriftDetections / status.totalDataChecks : 0.0);
        System.out.printf("  • Concept Checks: %d (Drift: %d, Rate: %.2f%%)\n", 
                         status.totalConceptChecks, status.conceptDriftDetections,
                         status.totalConceptChecks > 0 ? 100.0 * status.conceptDriftDetections / status.totalConceptChecks : 0.0);
        System.out.printf("  • Total Alerts: %d\n", status.totalAlerts);
        
        System.out.println("\n🎯 ACCURACY METRICS:");
        System.out.printf("  • Overall Accuracy: %.4f\n", status.overallAccuracy);
        System.out.printf("  • Window Accuracy: %.4f\n", status.slidingWindowAccuracy);
        
        if (status.lastAlertTime != null) {
            System.out.printf("\n🚨 Last Alert: %s\n", dateFormat.format(status.lastAlertTime));
        }
        
        System.out.printf("\n📝 Event History: %d events\n", status.eventHistorySize);
        
        // Show recent events
        List<DriftEvent> recentEvents = new ArrayList<>(eventHistory);
        if (!recentEvents.isEmpty()) {
            System.out.println("\n🕐 RECENT EVENTS:");
            recentEvents.stream()
                .skip(Math.max(0, recentEvents.size() - 5)) // Last 5 events
                .forEach(event -> {
                    String icon = event.getSeverity() == DriftEvent.Severity.CRITICAL ? "🚨" :
                                 event.getSeverity() == DriftEvent.Severity.WARNING ? "⚠️" : "ℹ️";
                    System.out.printf("  %s %s: %s\n", icon, 
                                     dateFormat.format(event.getTimestamp()), 
                                     event.getMessage());
                });
        }
        
        System.out.println("=".repeat(60) + "\n");
    }
    
    /**
     * Reset all monitoring state.
     */
    public void reset() {
        dataDriftDetector.reset();
        conceptDriftDetector.reset();
        eventHistory.clear();
        
        totalDataChecks = 0;
        totalConceptChecks = 0;
        dataDriftDetections = 0;
        conceptDriftDetections = 0;
        totalAlerts = 0;
        lastAlertTime = null;
        
        logEvent(new DriftEvent("RESET", "Dashboard and detectors reset", new Date(), DriftEvent.Severity.INFO));
    }
    
    /**
     * Log a drift event.
     */
    private void logEvent(DriftEvent event) {
        eventHistory.offer(event);
        
        // Maintain event history size
        while (eventHistory.size() > config.getMaxHistorySize()) {
            eventHistory.poll();
        }
        
        if (config.isDetailedLoggingEnabled()) {
            String icon = event.getSeverity() == DriftEvent.Severity.CRITICAL ? "🚨" :
                         event.getSeverity() == DriftEvent.Severity.WARNING ? "⚠️" : "ℹ️";
            System.out.printf("%s [%s] %s: %s\n", icon, event.getSeverity(), 
                             dateFormat.format(event.getTimestamp()), event.getMessage());
        }
    }
    
    /**
     * Trigger alert for drift event.
     */
    private void triggerAlert(DriftEvent event) {
        if (!config.isAlertsEnabled()) {
            return;
        }
        
        // Check cooldown period
        Date now = new Date();
        if (lastAlertTime != null && (now.getTime() - lastAlertTime.getTime()) < alertCooldownMs) {
            return; // Skip alert due to cooldown
        }
        
        totalAlerts++;
        lastAlertTime = now;
        
        // Print alert to console
        String icon = event.getSeverity() == DriftEvent.Severity.CRITICAL ? "🚨" : "⚠️";
        System.out.printf("\n%s DRIFT ALERT #%d %s\n", icon, totalAlerts, icon);
        System.out.printf("Time: %s\n", dateFormat.format(event.getTimestamp()));
        System.out.printf("Type: %s\n", event.getEventType());
        System.out.printf("Message: %s\n", event.getMessage());
        if (event.getFeatureName() != null) {
            System.out.printf("Feature: %s\n", event.getFeatureName());
        }
        if (event.getDriftScore() >= 0) {
            System.out.printf("Drift Score: %.4f\n", event.getDriftScore());
        }
        System.out.println("=" + "=".repeat(40) + "\n");
    }
    
    // Inner classes for dashboard data structures
    
    public static class DriftEvent {
        public enum Severity { INFO, WARNING, CRITICAL }
        
        private final String eventType;
        private final String message;
        private final Date timestamp;
        private final Severity severity;
        private String featureName;
        private double driftScore = -1.0;
        private String detectionMethod;
        
        public DriftEvent(String eventType, String message, Date timestamp, Severity severity) {
            this.eventType = eventType;
            this.message = message;
            this.timestamp = timestamp;
            this.severity = severity;
        }
        
        // Getters and setters
        public String getEventType() { return eventType; }
        public String getMessage() { return message; }
        public Date getTimestamp() { return timestamp; }
        public Severity getSeverity() { return severity; }
        public String getFeatureName() { return featureName; }
        public void setFeatureName(String featureName) { this.featureName = featureName; }
        public double getDriftScore() { return driftScore; }
        public void setDriftScore(double driftScore) { this.driftScore = driftScore; }
        public String getDetectionMethod() { return detectionMethod; }
        public void setDetectionMethod(String detectionMethod) { this.detectionMethod = detectionMethod; }
    }
    
    public static class MonitoringReport {
        public final Date startTime;
        public final Date endTime;
        public final long sessionDurationMs;
        public final int totalDataChecks;
        public final int totalConceptChecks;
        public final int totalAlerts;
        public final int dataDriftDetections;
        public final int conceptDriftDetections;
        public final double dataDriftRate;
        public final double conceptDriftRate;
        public final ConceptDriftDetector.DriftStatistics conceptStatistics;
        public final List<DriftEvent> events;
        public final DriftConfig configuration;
        
        public MonitoringReport(Date startTime, Date endTime, long sessionDurationMs,
                               int totalDataChecks, int totalConceptChecks, int totalAlerts,
                               int dataDriftDetections, int conceptDriftDetections,
                               double dataDriftRate, double conceptDriftRate,
                               ConceptDriftDetector.DriftStatistics conceptStatistics,
                               List<DriftEvent> events, DriftConfig configuration) {
            this.startTime = startTime;
            this.endTime = endTime;
            this.sessionDurationMs = sessionDurationMs;
            this.totalDataChecks = totalDataChecks;
            this.totalConceptChecks = totalConceptChecks;
            this.totalAlerts = totalAlerts;
            this.dataDriftDetections = dataDriftDetections;
            this.conceptDriftDetections = conceptDriftDetections;
            this.dataDriftRate = dataDriftRate;
            this.conceptDriftRate = conceptDriftRate;
            this.conceptStatistics = conceptStatistics;
            this.events = events;
            this.configuration = configuration;
        }
    }
    
    public static class DashboardStatus {
        public final boolean isMonitoring;
        public final Date monitoringStartTime;
        public final int totalDataChecks;
        public final int totalConceptChecks;
        public final int dataDriftDetections;
        public final int conceptDriftDetections;
        public final int totalAlerts;
        public final double overallAccuracy;
        public final double slidingWindowAccuracy;
        public final Date lastAlertTime;
        public final int eventHistorySize;
        
        public DashboardStatus(boolean isMonitoring, Date monitoringStartTime,
                              int totalDataChecks, int totalConceptChecks,
                              int dataDriftDetections, int conceptDriftDetections,
                              int totalAlerts, double overallAccuracy, double slidingWindowAccuracy,
                              Date lastAlertTime, int eventHistorySize) {
            this.isMonitoring = isMonitoring;
            this.monitoringStartTime = monitoringStartTime;
            this.totalDataChecks = totalDataChecks;
            this.totalConceptChecks = totalConceptChecks;
            this.dataDriftDetections = dataDriftDetections;
            this.conceptDriftDetections = conceptDriftDetections;
            this.totalAlerts = totalAlerts;
            this.overallAccuracy = overallAccuracy;
            this.slidingWindowAccuracy = slidingWindowAccuracy;
            this.lastAlertTime = lastAlertTime;
            this.eventHistorySize = eventHistorySize;
        }
    }
}
