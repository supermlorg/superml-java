# SuperML Drift Detection Module

The SuperML Drift Detection module provides comprehensive monitoring and detection of data drift and concept drift in machine learning models. This enterprise-grade solution enables real-time monitoring of model performance and data distribution changes in production environments.

## 🎯 Features

### Data Drift Detection
- **Kolmogorov-Smirnov Test**: Statistical test for comparing continuous distributions
- **Population Stability Index (PSI)**: Industry-standard metric for measuring distribution shifts
- **Jensen-Shannon Divergence**: Symmetric measure of similarity between probability distributions
- **Chi-Square Test**: Statistical test for categorical features
- **Statistical Moments Analysis**: Comparison of mean, standard deviation, skewness, and kurtosis

### Concept Drift Detection
- **Sliding Window Accuracy**: Monitor model performance over time
- **DDM (Drift Detection Method)**: Error rate-based drift detection
- **EDDM (Early Drift Detection Method)**: Distance-based early warning system
- **ADWIN (Adaptive Windowing)**: Adaptive window size for change detection
- **Confidence-based Detection**: Monitor prediction confidence patterns
- **Distribution Shift Detection**: Detect changes in prediction distributions

### Real-time Monitoring & Alerting
- **Interactive Dashboard**: Real-time monitoring with comprehensive statistics
- **Alert System**: Configurable alerts with cooldown periods
- **Event Logging**: Complete history of drift events and model performance
- **Export Capabilities**: JSON and CSV export for reporting and analysis

## 🚀 Quick Start

### Basic Usage

```java
import org.superml.drift.*;

// Create configuration
DriftConfig config = DriftConfig.sensitiveDetection();

// Initialize dashboard
DriftDashboard dashboard = new DriftDashboard(config);
dashboard.startMonitoring();

// Set baseline accuracy
dashboard.setBaselineAccuracy(0.85);

// Check for data drift
double[] referenceData = {1.0, 2.0, 3.0, 4.0, 5.0};
double[] currentData = {2.0, 3.0, 4.0, 5.0, 6.0};
DataDriftDetector.DataDriftResult dataDrift = 
    dashboard.checkDataDrift(referenceData, currentData, "feature_name");

// Check for concept drift
ConceptDriftDetector.ConceptDriftResult conceptDrift = 
    dashboard.checkConceptDrift(prediction, actualLabel, confidence);

// Print real-time dashboard
dashboard.printDashboard();

// Generate report
MonitoringReport report = dashboard.stopMonitoring();
```

### Advanced Configuration

```java
// Custom configuration using builder pattern
DriftConfig config = new DriftConfig.Builder()
    .ksSignificanceLevel(0.01)      // More strict KS test
    .psiThreshold(0.1)              // Lower PSI threshold
    .accuracyDropThreshold(0.02)    // Sensitive to accuracy drops
    .slidingWindowSize(500)         // Smaller window for faster detection
    .enableAlerts(true)             // Enable real-time alerts
    .outputFormat("JSON")           // JSON export format
    .build();

DriftDashboard dashboard = new DriftDashboard(config);
```

### Preset Configurations

```java
// Sensitive detection (lower thresholds, faster alerts)
DriftConfig sensitive = DriftConfig.sensitiveDetection();

// Conservative detection (higher thresholds, fewer false positives)
DriftConfig conservative = DriftConfig.conservativeDetection();

// Balanced detection (default settings)
DriftConfig balanced = DriftConfig.balancedDetection();

// Fast detection (smaller windows, quicker response)
DriftConfig fast = DriftConfig.fastDetection();

// Robust detection (larger windows, more stable)
DriftConfig robust = DriftConfig.robustDetection();
```

## 📊 Data Drift Detection Methods

### 1. Kolmogorov-Smirnov Test
Tests whether two samples come from the same continuous distribution.

```java
DataDriftDetector detector = new DataDriftDetector(config);
DataDriftResult result = detector.detectDrift(referenceData, currentData, "feature");

if (result.getKsTestResult().isDrift) {
    System.out.printf("KS drift detected: p-value = %.6f", 
                     result.getKsTestResult().pValue);
}
```

### 2. Population Stability Index (PSI)
Measures the shift in a variable's distribution.

- **PSI < 0.1**: No significant change
- **0.1 ≤ PSI < 0.2**: Moderate change
- **PSI ≥ 0.2**: Significant change

```java
if (result.getPsiTestResult().isDrift) {
    System.out.printf("PSI drift detected: PSI = %.4f", 
                     result.getPsiTestResult().psiValue);
}
```

### 3. Jensen-Shannon Divergence
Symmetric measure of similarity between probability distributions.

```java
if (result.getJsTestResult().isDrift) {
    System.out.printf("JS drift detected: divergence = %.4f", 
                     result.getJsTestResult().divergence);
}
```

## 🎯 Concept Drift Detection Methods

### 1. DDM (Drift Detection Method)
Monitors error rate and variance to detect performance degradation.

```java
ConceptDriftDetector detector = new ConceptDriftDetector(config);
detector.setBaselineAccuracy(0.85);

ConceptDriftResult result = detector.recordPrediction(prediction, actualLabel, confidence, timestamp);

if (result.isDriftDetected() && result.getDetectionMethod().equals("DDM")) {
    System.out.println("DDM detected concept drift!");
}
```

### 2. EDDM (Early Drift Detection Method)
Detects drift earlier by monitoring distance between classification errors.

### 3. ADWIN (Adaptive Windowing)
Uses adaptive window sizes to detect changes in data streams.

### 4. Sliding Window Accuracy
Monitors accuracy over a sliding window of recent predictions.

## 📈 Dashboard and Monitoring

### Real-time Dashboard

```java
dashboard.printDashboard();
```

Output:
```
============================================================
🎯 SUPERML DRIFT MONITORING DASHBOARD
============================================================
📊 Status: 🟢 MONITORING
⏰ Started: 2024-01-15 10:30:00

📈 STATISTICS:
  • Data Checks: 25 (Drift: 3, Rate: 12.00%)
  • Concept Checks: 500 (Drift: 2, Rate: 0.40%)
  • Total Alerts: 5

🎯 ACCURACY METRICS:
  • Overall Accuracy: 0.8420
  • Window Accuracy: 0.8150

🚨 Last Alert: 2024-01-15 11:45:23

📝 Event History: 47 events
============================================================
```

### Export Reports

```java
// Export detailed JSON report
MonitoringReport report = dashboard.generateReport(new Date());
dashboard.exportReportToJson(report, "drift_report.json");

// Export events to CSV
dashboard.exportEventsToCSV("drift_events.csv");
```

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ksSignificanceLevel` | 0.05 | P-value threshold for KS test |
| `psiThreshold` | 0.2 | PSI value threshold for drift detection |
| `jsThreshold` | 0.1 | Jensen-Shannon divergence threshold |
| `accuracyDropThreshold` | 0.05 | Minimum accuracy drop to trigger drift |
| `slidingWindowSize` | 1000 | Size of sliding window for accuracy monitoring |
| `confidenceDropThreshold` | 0.1 | Threshold for confidence-based drift |
| `enableAlerts` | true | Enable real-time alerting |

## 🔧 Integration Examples

### With Existing ML Pipeline

```java
public class MLPipeline {
    private DriftDashboard driftMonitor;
    private Model model;
    
    public void initializeMonitoring() {
        DriftConfig config = DriftConfig.balancedDetection();
        driftMonitor = new DriftDashboard(config);
        driftMonitor.startMonitoring();
        driftMonitor.setBaselineAccuracy(model.getValidationAccuracy());
    }
    
    public Prediction predict(Features features) {
        Prediction pred = model.predict(features);
        
        // Monitor for drift
        if (hasGroundTruth()) {
            driftMonitor.checkConceptDrift(pred.getValue(), 
                                         getGroundTruth(), 
                                         pred.getConfidence());
        }
        
        return pred;
    }
    
    public void checkFeatureDrift(String featureName, double[] newData) {
        double[] referenceData = getReferenceData(featureName);
        driftMonitor.checkDataDrift(referenceData, newData, featureName);
    }
}
```

### Batch Processing

```java
// Process batch of predictions
double[] predictions = {0.8, 0.6, 0.9, 0.7};
Double[] actualLabels = {0.8, 0.5, 0.9, 0.8};
double[] confidences = {0.85, 0.75, 0.95, 0.80};

ConceptDriftDetector.BatchDriftResult batchResult = 
    conceptDetector.processBatch(predictions, actualLabels, confidences);

System.out.printf("Batch processed: %d drift detections out of %d predictions\n", 
                 batchResult.getDriftDetections(), predictions.length);
```

## 📊 Performance Metrics

The drift detection system tracks comprehensive metrics:

- **Detection Accuracy**: How accurately the system detects true drift
- **False Positive Rate**: Rate of false drift alerts
- **Detection Latency**: Time between drift occurrence and detection
- **Computational Overhead**: Processing time per prediction/batch

## 🔬 Statistical Methods Details

### Kolmogorov-Smirnov Test
- **Use Case**: Continuous features
- **Null Hypothesis**: Both samples come from the same distribution
- **Statistic**: Maximum difference between cumulative distributions
- **Interpretation**: p-value < significance level indicates drift

### Population Stability Index
- **Formula**: PSI = Σ((%Expected - %Actual) × ln(%Expected / %Actual))
- **Binning**: Automatic percentile-based binning for continuous features
- **Interpretation**: Higher PSI indicates more distribution shift

### Jensen-Shannon Divergence
- **Range**: [0, 1] where 0 = identical distributions, 1 = completely different
- **Symmetric**: JS(P||Q) = JS(Q||P)
- **Smoothing**: Automatic handling of zero probabilities

## 🚨 Alert System

### Alert Types
- **Data Drift**: Feature distribution changes
- **Concept Drift**: Model performance degradation  
- **Warning**: Early indicators of potential drift
- **Critical**: Immediate attention required

### Alert Configuration
```java
DriftConfig config = new DriftConfig.Builder()
    .enableAlerts(true)
    .enableDetailedLogging(true)
    .build();
```

### Alert Cooldown
Prevents alert spam with configurable cooldown periods (default: 1 minute).

## 📝 Best Practices

1. **Baseline Selection**: Use representative training/validation data as reference
2. **Threshold Tuning**: Start with preset configurations and adjust based on domain
3. **Multi-method Approach**: Use multiple detection methods for robust monitoring
4. **Regular Retraining**: Reset baselines after model retraining
5. **Feature-wise Monitoring**: Monitor each feature separately for detailed insights
6. **Gradual Deployment**: Start with conservative thresholds in production

## 🛠️ Dependencies

- Apache Commons Math 3.6.1 (statistical functions)
- Jackson 2.15.2 (JSON processing)
- Apache Commons CSV 1.9.0 (CSV export)

## 📚 References

1. Gama, J. et al. "Learning with Drift Detection" (2004)
2. Baena-García, M. et al. "Early Drift Detection Method" (2006)  
3. Bifet, A. & Gavaldà, R. "Learning from Time-Changing Data with Adaptive Windowing" (2007)
4. Narasimhan, B. et al. "On the Challenge of Model Monitoring" (2018)

## 🤝 Contributing

Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to the SuperML project.

## 📄 License

This module is part of the SuperML framework and is licensed under the Apache License 2.0.
