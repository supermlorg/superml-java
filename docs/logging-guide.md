---
title: "Logging Configuration Guide"
description: "Comprehensive guide to logging configuration and best practices in SuperML Java"
layout: default
toc: true
search: true
---

# SuperML Java - Logging Configuration

SuperML Java now uses **Logback** with **SLF4J** for professional logging, providing structured, configurable, and production-ready logging capabilities.

## Logging Features

### 🎯 **Multiple Output Formats**
- **Console**: Colored output for development
- **File**: Structured file logging with rotation
- **JSON**: Machine-readable structured logs (optional)

### 📊 **Log Levels**
- `ERROR`: Critical errors requiring immediate attention
- `WARN`: Warning conditions
- `INFO`: General information about application flow
- `DEBUG`: Detailed debugging information
- `TRACE`: Very detailed debugging information

### 🎨 **Colored Console Output**
- Colored log levels for better readability
- Thread names and logger names highlighted
- Timestamps formatted for easy reading

## Configuration

### Default Configuration
The framework comes with a pre-configured `logback.xml` that provides:

```xml
<!-- Console with colors -->
%d{yyyy-MM-dd HH:mm:ss.SSS} %highlight(%-5level) %cyan([%thread]) %green(%logger{36}) - %msg%n

<!-- File logging with rotation -->
logs/superml.log (rotates daily, keeps 30 days, max 1GB total)
```

### Component-Specific Logging
Different components have tailored log levels:

- **KaggleIntegration**: `DEBUG` level for API calls and downloads
- **KaggleTrainingManager**: `DEBUG` level for training workflows  
- **HTTP Clients**: `WARN` level to reduce noise
- **Root Logger**: `INFO` level for general framework usage

### File Structure
```
logs/
├── superml.log              # Current log file
├── superml.2024-01-15.log   # Archived daily logs
└── superml-json.log         # JSON format logs (if enabled)
```

## Usage Examples

### In Your Code
```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyClass {
    private static final Logger logger = LoggerFactory.getLogger(MyClass.class);
    
    public void myMethod() {
        logger.info("Starting data processing for dataset: {}", datasetName);
        logger.debug("Processing {} rows with {} features", rows, features);
        
        try {
            // Your code here
            logger.info("Processing completed successfully");
        } catch (Exception e) {
            logger.error("Processing failed for dataset: {}", datasetName, e);
        }
    }
}
```

### Kaggle Integration Logging
```java
KaggleIntegration kaggle = new KaggleIntegration(credentials, "datasets");
// Logs: "INFO  - Downloading dataset: owner/dataset-name"
// Logs: "DEBUG - Downloaded: /path/to/file.csv"

KaggleTrainingManager trainer = new KaggleTrainingManager(kaggle);
// Logs: "INFO  - === Kaggle Training Manager ==="
// Logs: "INFO  - Training LogisticRegression..."
// Logs: "INFO  - Best model: LogisticRegression (Score: 0.9542)"
```

## Environment-Specific Configuration

### Development Profile
```xml
<springProfile name="dev">
    <root level="DEBUG">
        <appender-ref ref="CONSOLE"/>
    </root>
</springProfile>
```

### Production Profile  
```xml
<springProfile name="prod">
    <root level="WARN">
        <appender-ref ref="FILE"/>
        <appender-ref ref="JSON_FILE"/>
    </root>
</springProfile>
```

## Advanced Configuration

### Custom Log Levels
Modify `logback.xml` to change log levels for specific components:

```xml
<!-- More verbose Kaggle logging -->
<logger name="org.superml.datasets.KaggleIntegration" level="TRACE"/>

<!-- Quiet HTTP client -->
<logger name="org.apache.hc" level="ERROR"/>
```

### Custom Appenders
Add file appenders for specific components:

```xml
<appender name="KAGGLE_FILE" class="ch.qos.logback.core.FileAppender">
    <file>logs/kaggle-integration.log</file>
    <encoder>
        <pattern>%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n</pattern>
    </encoder>
</appender>

<logger name="org.superml.datasets" level="DEBUG" additivity="false">
    <appender-ref ref="KAGGLE_FILE"/>
</logger>
```

## Migration from System.out.println

The framework has been updated to replace `System.out.println` with appropriate logging:

### Before
```java
System.out.println("Downloading dataset: " + owner + "/" + name);
System.out.println("Training completed with score: " + score);
```

### After  
```java
logger.info("Downloading dataset: {}/{}", owner, name);
logger.info("Training completed with score: {:.4f}", score);
```

### Benefits
- **Structured**: Parameterized messages prevent string concatenation
- **Configurable**: Can be turned on/off without code changes
- **Searchable**: JSON logs enable powerful search and analysis
- **Performance**: Lazy evaluation of log messages
- **Professional**: Production-ready logging standards

## Best Practices

### 1. Use Appropriate Log Levels
```java
logger.error("Failed to connect to Kaggle API", exception);  // Critical errors
logger.warn("Dataset download is taking longer than expected");  // Warnings
logger.info("Training completed successfully");  // General information
logger.debug("Feature preprocessing took {} ms", duration);  // Debug details
```

### 2. Use Parameterized Messages
```java
// ✅ Good: Parameterized
logger.info("Processing dataset {} with {} samples", name, count);

// ❌ Avoid: String concatenation  
logger.info("Processing dataset " + name + " with " + count + " samples");
```

### 3. Include Context
```java
logger.info("Starting Kaggle training for dataset: {}, target: {}, algorithms: {}", 
    datasetName, targetColumn, Arrays.toString(algorithms));
```

### 4. Log Exceptions Properly
```java
try {
    // risky operation
} catch (Exception e) {
    logger.error("Operation failed for dataset: {}", datasetName, e);
}
```

This logging framework provides the SuperML Java framework with enterprise-grade logging capabilities while maintaining simplicity for development and debugging.
