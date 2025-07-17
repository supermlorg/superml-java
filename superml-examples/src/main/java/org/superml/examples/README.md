# SuperML Java Examples

This directory contains comprehensive examples demonstrating the SuperML Java framework capabilities. These examples are designed for learning, testing, and integration into your own projects.

## 📁 Example Files

### 🌸 [BasicClassification.java](BasicClassification.java)
**Complete classification workflow using synthetic data**
- Generates classification dataset similar to Iris (150 samples, 4 features, 3 classes)
- Train/test splitting and model training
- Performance metrics and confusion matrix
- Sample predictions display with detailed output

**Features demonstrated:**
- `LogisticRegression` classifier
- `Datasets.makeClassification()` for synthetic data generation
- `ModelSelection.trainTestSplit()` for data splitting
- `Metrics` for evaluation (accuracy, precision, recall, F1-score)

---

### 📊 [RegressionComparison.java](RegressionComparison.java)
**Comprehensive regression analysis comparing multiple algorithms**
- Linear Regression baseline model
- Ridge Regression with L2 regularization
- Lasso Regression with L1 regularization and feature selection
- Performance comparison and feature analysis
- Model coefficient inspection

**Features demonstrated:**
- `LinearRegression`, `Ridge`, `Lasso` models
- `Datasets.makeRegression()` for synthetic regression data
- Regularization parameter tuning and comparison
- Feature coefficient analysis and importance
- Cross-validation and model selection

---

### 🔧 [PipelineExample.java](PipelineExample.java)
**ML Pipeline for chaining preprocessing and models**
- Data standardization with `StandardScaler`
- Pipeline creation and training workflow
- Feature scaling statistics and analysis
- Model coefficient analysis
- End-to-end workflow automation

**Features demonstrated:**
- `Pipeline` for workflow automation and reproducibility
- `StandardScaler` for feature preprocessing and normalization
- Pipeline step management and introspection
- Feature transformation inspection and validation
- Integrated model evaluation within pipelines

---

### ⚡ [InferenceExample.java](InferenceExample.java)
**Production-ready inference system**
- Real-time single predictions with microsecond timing
- Asynchronous prediction processing with `CompletableFuture`
- Batch inference for high-throughput scenarios
- Performance monitoring and caching
- Model loading, persistence, and lifecycle management

**Features demonstrated:**
- `InferenceEngine` for production deployment
- `BatchInferenceProcessor` for high-throughput processing
- Async prediction with thread-safe operations
- Performance metrics collection and monitoring
- Model warm-up optimization and caching strategies

---

### 🏆 [KaggleIntegration.java](KaggleIntegration.java)
**Complete competition workflow for data science competitions**
- Data exploration and statistical analysis
- Cross-validation for robust model evaluation
- Feature importance analysis and selection
- Competition submission generation
- Model persistence and deployment preparation

**Features demonstrated:**
- 5-fold cross-validation with statistical analysis
- Feature importance visualization and interpretation
- Competition metrics calculation and optimization
- Automated submission file generation
- Robust model evaluation and validation strategies

---

### 🚀 [BasicXGBoostExample.java](BasicXGBoostExample.java)
**XGBoost quick start and essential features**
- Basic XGBoost training with default parameters
- Advanced configuration with regularization and early stopping
- Feature importance analysis with multiple metrics
- Hyperparameter tuning best practices
- Performance optimization and validation

**Features demonstrated:**
- `XGBoost` classifier with gradient boosting
- L1/L2 regularization (Alpha/Lambda parameters)
- Early stopping with validation data
- Feature importance analysis (weight, gain, cover)
- Tree pruning with gamma parameter
- Row and column subsampling for overfitting prevention

---

### 🌟 [XGBoostExample.java](XGBoostExample.java)
**Comprehensive XGBoost showcase for competition-level ML**
- Advanced hyperparameter tuning and optimization
- Model comparison with Random Forest and Gradient Boosting
- Feature importance analysis with visualization
- Production-ready pipeline integration
- Cross-validation and performance monitoring
- Inference performance testing and deployment preparation

**Features demonstrated:**
- Advanced XGBoost configuration with all parameters
- Histogram-based approximate split finding
- Parallel tree construction and optimization
- Feature engineering through gradient boosting
- Model ensemble comparison and benchmarking
- Production pipeline with preprocessing integration
- Competition-ready workflow and best practices

## 🚀 Running the Examples

### Prerequisites
- Java 11 or higher
- Maven (for dependency management)
- SuperML Java framework compiled

### Compilation and Execution

```bash
# Navigate to the SuperML Java root directory
cd /path/to/superml-java

# Compile the main framework
mvn compile

# Compile examples
javac -cp "target/classes" examples/*.java

# Run any example (replace with desired example)
java -cp "target/classes:examples" examples.BasicClassification
java -cp "target/classes:examples" examples.RegressionComparison
java -cp "target/classes:examples" examples.PipelineExample
java -cp "target/classes:examples" examples.InferenceExample
java -cp "target/classes:examples" examples.KaggleIntegration
java -cp "target/classes:examples" examples.BasicXGBoostExample
java -cp "target/classes:examples" examples.XGBoostExample
```

### Alternative: Maven Execution

```bash
# Run examples directly with Maven
mvn exec:java -Dexec.mainClass="examples.BasicClassification" -Dexec.args="" -Dexec.classpathScope="compile"
```

## 📖 Example Output

Each example produces detailed console output showing:
- -> Step-by-step progress indicators
- 📊 Performance metrics and statistics
- 🎯 Prediction results and analysis
- 💡 Educational tips and insights
- ⚡ Timing and performance information

### Sample Output Structure:
```
============================================================
           SuperML Java - [Example Name]
============================================================
Generating synthetic dataset...
✓ Dataset generated: 150 samples, 4 features, 3 classes

Training model...
✓ Model trained in 45 ms

Performance Metrics:
============================================================
Accuracy:        0.9556 (95.6%)
Precision:       0.9583
Recall:          0.9556
F1-Score:        0.9563

Sample Predictions:
============================================================
Actual | Predicted | Class Name
  0    |     0     | Class A ✓
  1    |     1     | Class B ✓
  2    |     2     | Class C ✓

✓ Example completed successfully!
💡 [Educational insight about the demonstrated features]
```

## 🎓 Learning Path

**Recommended order for beginners:**

1. **BasicClassification.java** - Start here for fundamental ML concepts
2. **RegressionComparison.java** - Learn about different regression algorithms
3. **PipelineExample.java** - Understand workflow automation and preprocessing
4. **BasicXGBoostExample.java** - Learn XGBoost fundamentals and gradient boosting
5. **InferenceExample.java** - Production deployment and performance optimization
6. **XGBoostExample.java** - Advanced XGBoost techniques and competition workflows
7. **KaggleIntegration.java** - Complete competition strategies and best practices

## 🔗 Integration with Documentation

These examples are referenced in the [SuperML Java Documentation](https://supermlorg.github.io/superml-java/) and provide hands-on experience with the concepts covered in:

- [Quick Start Guide](../docs/quick-start.md)
- [API Documentation](../docs/api/)
- [Implementation Summary](../docs/implementation-summary.md)
- [Inference Guide](../docs/inference-guide.md)
- [Kaggle Integration Guide](../docs/kaggle-integration.md)

## 💡 Tips for Modification

- **Dataset Customization**: Replace synthetic data generators with your own data loading logic
- **Model Tuning**: Experiment with different hyperparameters and algorithms
- **Feature Engineering**: Add custom preprocessing steps in the pipeline examples
- **Performance Optimization**: Modify batch sizes, caching parameters, and threading in inference examples
- **Evaluation Metrics**: Add domain-specific metrics relevant to your use case
- **Competition Strategy**: Adapt the Kaggle example for your specific competition requirements

## 🛠️ Troubleshooting

### Common Issues:

**Classpath Problems:**
```bash
# Ensure all dependencies are compiled
mvn clean compile

# Verify classpath includes both framework and examples
java -cp "target/classes:examples" examples.YourExample
```

**Memory Issues with Large Datasets:**
```bash
# Increase JVM memory for large datasets
export MAVEN_OPTS="-Xmx4g"
java -Xmx4g -cp "target/classes:examples" examples.YourExample
```

**Missing Dependencies:**
```bash
# Install framework dependencies
mvn clean install

# Verify all required packages are available
mvn dependency:tree
```

**Performance Issues:**
```bash
# Enable performance optimizations
java -server -XX:+UseG1GC -cp "target/classes:examples" examples.InferenceExample
```

## 📚 Additional Resources

- [GitHub Repository](https://github.com/supermlorg/superml-java)
- [Documentation Site](https://supermlorg.github.io/superml-java/)
- [API Reference](https://supermlorg.github.io/superml-java/api/)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Issues and Support](https://github.com/supermlorg/superml-java/issues)

## 🤝 Contributing

Want to add your own example? We welcome contributions! Please:

1. Follow the existing code style and documentation format
2. Include comprehensive comments and educational insights
3. Add performance timing and metrics where applicable
4. Test your example thoroughly with different scenarios
5. Update this README with your new example description

---

**Happy Learning with SuperML Java! 🚀**

*Ready for production ML workflows, competition success, and educational exploration.*

### Alternative: Direct Compilation

```bash
# Compile with classpath
javac -cp "target/classes:ml-framework/target/classes" examples/*.java

# Run examples
java -cp ".:target/classes:ml-framework/target/classes" examples.BasicClassification
```

## 📖 Example Output

Each example produces detailed console output showing:
- -> Step-by-step progress indicators
- 📊 Performance metrics and statistics
- 🎯 Prediction results and analysis
- 💡 Educational tips and insights
- ⚡ Timing and performance information

### Sample Output Structure:
```
============================================================
           SuperML Java - [Example Name]
============================================================
Loading dataset...
✓ Dataset loaded: 150 samples, 4 features, 3 classes

Training model...
✓ Model trained in 45 ms

Performance Metrics:
============================================================
Accuracy:        0.9556 (95.6%)
Precision:       0.9583
Recall:          0.9556
F1-Score:        0.9563

✓ Example completed successfully!
💡 [Educational insight about the demonstrated features]
```

## 🎓 Learning Path

**Recommended order for beginners:**

1. **BasicClassification.java** - Start here for fundamental concepts
2. **RegressionComparison.java** - Learn about different algorithms
3. **PipelineExample.java** - Understand workflow automation
4. **InferenceExample.java** - Production deployment patterns
5. **KaggleIntegration.java** - Competition and advanced techniques

## 🔗 Integration with Documentation

These examples are referenced in the [SuperML Java Documentation](https://supermlorg.github.io/superml-java/) and provide hands-on experience with the concepts covered in:

- [Quick Start Guide](../docs/quick-start.md)
- [API Documentation](../docs/api/)
- [Implementation Summary](../docs/implementation-summary.md)
- [Inference Guide](../docs/inference-guide.md)

## 💡 Tips for Modification

- **Dataset Customization**: Replace `Datasets.loadIris()` with your own data loading logic
- **Model Tuning**: Experiment with different hyperparameters in each example
- **Feature Engineering**: Add preprocessing steps in the pipeline examples
- **Performance Optimization**: Modify batch sizes and caching parameters in inference examples
- **Evaluation Metrics**: Add custom metrics relevant to your use case

## 🛠️ Troubleshooting

### Common Issues:

**Classpath Problems:**
```bash
# Ensure all dependencies are compiled
mvn clean compile
```

**Memory Issues with Large Datasets:**
```bash
# Increase JVM memory
export MAVEN_OPTS="-Xmx2g"
mvn exec:java -Dexec.mainClass="examples.YourExample"
```

**Missing Dependencies:**
```bash
# Install framework dependencies
mvn install
```

## 📚 Additional Resources

- [GitHub Repository](https://github.com/supermlorg/superml-java)
- [Documentation Site](https://supermlorg.github.io/superml-java/)
- [API Reference](https://supermlorg.github.io/superml-java/api/)
- [Contributing Guidelines](../CONTRIBUTING.md)

---

**Happy Learning with SuperML Java! 🚀**
