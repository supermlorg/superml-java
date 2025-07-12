# SuperML Java Examples

**⚠️ NOTICE: Examples have been moved to the main source tree**

The examples in this directory are now deprecated. The updated examples are located in the main source tree at:

```
src/main/java/com/superml/examples/
```

## 📁 Updated Example Files

### 🌸 [BasicClassification.java](../src/main/java/com/superml/examples/BasicClassification.java)
**Complete classification workflow using synthetic data**
- Generates classification dataset similar to Iris
- Train/test splitting and model training
- Performance metrics and confusion matrix
- Sample predictions display

### 📊 [RegressionExample.java](../src/main/java/com/superml/examples/RegressionExample.java)
**Linear regression analysis with synthetic data**
- Generates regression dataset  
- Model training and evaluation
- Performance metrics (R², RMSE, MAE)
- Feature coefficient analysis

### 🔧 [PipelineExample.java](../src/main/java/com/superml/examples/PipelineExample.java)
**ML Pipeline for chaining preprocessing and models**
- Data standardization with `StandardScaler`
- Pipeline creation and training
- Feature scaling statistics
- End-to-end workflow automation

### ⚡ [InferenceExample.java](../src/main/java/com/superml/examples/InferenceExample.java)
**Production-ready inference system**
- Real-time single predictions
- Asynchronous prediction processing
- Batch inference processing
- Performance monitoring and metrics

## 🚀 Running the Updated Examples

### Prerequisites
- Java 11 or higher
- Maven (for dependency management)
- SuperML Java framework

### Compilation and Execution

```bash
# Navigate to the SuperML Java root directory
cd /path/to/superml-java

# Compile the framework
mvn compile

# Run any example (replace with desired example)
mvn exec:java -Dexec.mainClass="com.superml.examples.BasicClassification"
mvn exec:java -Dexec.mainClass="com.superml.examples.RegressionExample"
mvn exec:java -Dexec.mainClass="com.superml.examples.PipelineExample"
mvn exec:java -Dexec.mainClass="com.superml.examples.InferenceExample"
```

### Alternative: Direct Compilation

```bash
# Compile with classpath
javac -cp "target/classes:ml-framework/target/classes" examples/*.java

# Run examples
java -cp ".:target/classes:ml-framework/target/classes" examples.BasicClassification
```

## 📖 Example Output

Each example produces detailed console output showing:
- ✅ Step-by-step progress indicators
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
