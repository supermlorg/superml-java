# SuperML Java Framework - Modular Architecture

This document describes the comprehensive modular architecture implemented for the SuperML Java framework.

## Overview

The SuperML Java framework has been restructured into a sophisticated 21-module Maven multi-module project to provide maximum flexibility, maintainability, and modularity. Users can include only the components they need, creating lightweight applications or comprehensive ML pipelines.

## 🏗️ Complete Module Structure (21 Modules)

### **Core Foundation Modules**

#### 1. **superml-core**
- **Description**: Foundation interfaces and base classes for the entire framework
- **Contains**: 
  - `BaseEstimator.java` - Base class for all estimators
  - `Classifier.java` - Classification interface
  - `Estimator.java` - Core estimator interface
  - `Regressor.java` - Regression interface
  - `SupervisedLearner.java` - Supervised learning base
  - `UnsupervisedLearner.java` - Unsupervised learning base
- **Dependencies**: None (foundation module)
- **Usage**: Required by all other modules
- **Size**: ~6 core interfaces and base classes

#### 2. **superml-utils**
- **Description**: Common utilities and helper functions
- **Contains**: 
  - Array manipulation utilities
  - Mathematical helper functions
  - Common validation methods
  - Type conversion utilities
- **Dependencies**: superml-core
- **Usage**: Shared utilities across all modules

### **Algorithm Implementation Modules**

#### 3. **superml-linear-models**
- **Description**: Linear regression and classification algorithms
- **Contains**: 
  - `LinearRegression.java` - OLS regression with closed-form solution
  - `LogisticRegression.java` - Binary and multiclass classification
  - `Ridge.java` - L2 regularized regression
  - `Lasso.java` - L1 regularized regression with coordinate descent
  - `SGDClassifier.java` - Stochastic gradient descent classification
  - `SGDRegressor.java` - Stochastic gradient descent regression
- **Dependencies**: superml-core, superml-utils, commons-math3
- **Algorithms**: 6 linear algorithms
- **Features**: Regularization, multiclass support, gradient-based optimization

#### 4. **superml-tree-models**
- **Description**: Decision trees and ensemble algorithms
- **Contains**:
  - `DecisionTreeClassifier.java` - CART implementation for classification
  - `DecisionTreeRegressor.java` - CART implementation for regression
  - `RandomForestClassifier.java` - Bootstrap aggregating ensemble
  - `RandomForestRegressor.java` - Ensemble regression
  - `GradientBoostingClassifier.java` - Gradient boosting implementation
  - `XGBoost.java` - Extreme Gradient Boosting implementation
- **Dependencies**: superml-core, superml-utils, commons-math3
- **Algorithms**: 6 tree-based algorithms
- **Features**: CART splitting, bootstrap sampling, gradient boosting, XGBoost

#### 5. **superml-neural**
- **Description**: Deep learning and neural network algorithms
- **Contains**:
  - `MLPClassifier.java` - Multi-Layer Perceptron for classification
  - `CNNClassifier.java` - Convolutional Neural Network for image processing
  - `RNNClassifier.java` - Recurrent Neural Network for sequence processing
  - Neural network utilities and optimizers
  - Activation functions and layer implementations
- **Dependencies**: superml-core, superml-utils, commons-math3
- **Algorithms**: 3 neural network algorithms
- **Features**: Deep learning, backpropagation, GPU acceleration ready

#### 6. **superml-clustering**
- **Description**: Unsupervised clustering algorithms
- **Contains**:
  - `KMeans.java` - K-means with k-means++ initialization
  - Cluster evaluation metrics
  - Initialization strategies
- **Dependencies**: superml-core, superml-utils, commons-math3
- **Algorithms**: 1 clustering algorithm
- **Features**: Multiple initialization methods, convergence criteria

### **Data Processing and Preparation Modules**

#### 7. **superml-preprocessing**
- **Description**: Data preprocessing and feature engineering
- **Contains**:
  - `StandardScaler.java` - Feature standardization and normalization
  - `MinMaxScaler.java` - Min-max scaling
  - `RobustScaler.java` - Robust scaling using median and IQR
  - `LabelEncoder.java` - Categorical variable encoding
  - Data transformation utilities
- **Dependencies**: superml-core, superml-utils
- **Features**: Multiple scaling strategies, categorical encoding, missing value handling

#### 7. **superml-datasets**
- **Description**: Dataset loading, generation, and management
- **Contains**:
  - `Datasets.java` - Built-in dataset loading (Iris, Wine, etc.)
  - Synthetic data generation (`makeClassification`, `makeRegression`)
  - CSV loading utilities
  - Data splitting and sampling methods
- **Dependencies**: superml-core, superml-utils, commons-csv
- **Features**: Built-in datasets, synthetic data generation, file I/O

#### 8. **superml-model-selection**
- **Description**: Model selection, validation, and hyperparameter tuning
- **Contains**:
  - `GridSearchCV.java` - Exhaustive grid search
  - `RandomizedSearchCV.java` - Randomized parameter search
  - `CrossValidation.java` - K-fold cross-validation
  - `TrainTestSplit.java` - Data splitting utilities
  - Parameter space definitions
- **Dependencies**: superml-core, superml-utils, superml-metrics
- **Features**: Parallel hyperparameter tuning, custom parameter spaces, statistical validation

### **Pipeline and Workflow Modules**

#### 9. **superml-pipeline**
- **Description**: ML pipelines for chaining preprocessing and models
- **Contains**:
  - `Pipeline.java` - Sequential step execution
  - `FeatureUnion.java` - Parallel feature combination
  - Pipeline step management
  - Workflow serialization
- **Dependencies**: superml-core, superml-preprocessing, superml-utils
- **Features**: Step chaining, parameter propagation, pipeline introspection

#### 10. **superml-autotrainer**
- **Description**: Automated machine learning and algorithm selection
- **Contains**:
  - `AutoTrainer.java` - Automated model selection and training
  - `AlgorithmSelector.java` - Intelligent algorithm recommendation
  - `HyperparameterOptimizer.java` - Advanced optimization strategies
  - `EnsembleBuilder.java` - Automatic ensemble construction
- **Dependencies**: superml-core, superml-linear-models, superml-tree-models, superml-model-selection
- **Features**: AutoML workflows, algorithm comparison, ensemble methods

### **Evaluation and Metrics Modules**

#### 11. **superml-metrics**
- **Description**: Comprehensive evaluation metrics for all ML tasks
- **Contains**:
  - `ClassificationMetrics.java` - Accuracy, precision, recall, F1-score
  - `RegressionMetrics.java` - MSE, MAE, R², RMSE
  - `ClusteringMetrics.java` - Silhouette score, inertia
  - `ConfusionMatrix.java` - Confusion matrix generation and analysis
  - Statistical significance testing
- **Dependencies**: superml-core, superml-utils
- **Features**: Complete metric suite, statistical analysis, visualization support

#### 12. **superml-visualization**
- **Description**: Advanced dual-mode visualization system (ASCII + XChart GUI)
- **Contains**:
  - `VisualizationFactory.java` - Dual-mode visualization creation
  - `XChartConfusionMatrix.java` - Professional GUI confusion matrix
  - `XChartScatterPlot.java` - Interactive scatter plots
  - `XChartClusterPlot.java` - Cluster visualization
  - ASCII fallback visualizations
- **Dependencies**: superml-core, superml-metrics, XChart (optional)
- **Features**: Professional GUI charts, ASCII fallback, automatic mode selection

### **Production and Deployment Modules**

#### 13. **superml-inference**
- **Description**: High-performance production inference engine
- **Contains**:
  - `InferenceEngine.java` - Optimized prediction serving
  - `BatchInferenceProcessor.java` - High-throughput batch processing
  - `ModelCache.java` - Intelligent model caching
  - `PredictionMonitor.java` - Performance monitoring
  - Async prediction capabilities
- **Dependencies**: superml-core, superml-utils, superml-metrics
- **Features**: Microsecond inference, caching, monitoring, async processing

#### 14. **superml-persistence**
- **Description**: Model serialization and lifecycle management
- **Contains**:
  - `ModelPersistence.java` - Save/load trained models
  - `ModelManager.java` - Model lifecycle management
  - `TrainingStatistics.java` - Automatic statistics capture
  - Metadata management
- **Dependencies**: superml-core, superml-utils, Jackson
- **Features**: Model versioning, automatic statistics, metadata tracking

### **External Integration Modules**

#### 15. **superml-kaggle**
- **Description**: Kaggle competition integration and automation
- **Contains**:
  - `KaggleClient.java` - Direct Kaggle API integration
  - `KaggleTrainingManager.java` - Automated competition workflows
  - `DatasetDownloader.java` - Automatic dataset management
  - Competition submission utilities
- **Dependencies**: superml-core, superml-autotrainer, Apache HttpClient
- **Features**: One-line training, automatic submissions, dataset management

#### 16. **superml-onnx**
- **Description**: ONNX model export and interoperability
- **Contains**:
  - `ONNXExporter.java` - Convert models to ONNX format
  - `ONNXModelAdapter.java` - ONNX model integration
  - Cross-platform model deployment
- **Dependencies**: superml-core, ONNX Runtime
- **Features**: Cross-platform deployment, model standardization

#### 17. **superml-pmml**
- **Description**: PMML (Predictive Model Markup Language) support
- **Contains**:
  - `PMMLExporter.java` - Export models to PMML format
  - `PMMLImporter.java` - Import PMML models
  - Industry-standard model exchange
- **Dependencies**: superml-core, JPMML libraries
- **Features**: Industry-standard model exchange, enterprise integration

#### 18. **superml-drift**
- **Description**: Model drift detection and monitoring
- **Contains**:
  - `DriftDetector.java` - Statistical drift detection
  - `DataDriftMonitor.java` - Feature drift monitoring
  - `ModelPerformanceTracker.java` - Performance degradation detection
  - Alert and notification systems
- **Dependencies**: superml-core, superml-metrics, superml-utils
- **Features**: Real-time monitoring, statistical tests, automated alerts

### **Packaging and Distribution Modules**

#### 19. **superml-bundle-all**
- **Description**: Complete framework distribution package
- **Contains**:
  - All-in-one dependency bundle
  - Complete feature set packaging
  - Simplified distribution
- **Dependencies**: All SuperML modules
- **Features**: Single dependency convenience, complete functionality

#### 20. **superml-examples**
- **Description**: Comprehensive example and demonstration suite
- **Contains**:
  - 11 complete examples covering all framework features
  - `XChartVisualizationExample.java` - Professional GUI showcase
  - `BasicClassification.java` - Fundamental ML concepts
  - `AutoMLExample.java` - Automated machine learning
  - Documentation and learning materials
- **Dependencies**: All SuperML modules for comprehensive demonstrations
- **Features**: Educational examples, GUI demonstrations, complete workflows

#### 21. **superml-java-parent**
- **Description**: Maven parent POM for unified build management
- **Contains**:
  - Centralized dependency management
  - Build configuration and standards
  - Version coordination across modules
  - Plugin management
- **Dependencies**: None (parent coordinator)
- **Features**: Unified builds, dependency coordination, release management

## 📊 Architecture Summary

### **Module Count and Distribution**
- **Total Modules**: 21 comprehensive modules
- **Core Foundation**: 2 modules (core, utils)
- **Algorithm Implementation**: 3 modules (linear-models, tree-models, clustering)
- **Data Processing**: 3 modules (preprocessing, datasets, model-selection)
- **Workflow Management**: 2 modules (pipeline, autotrainer)
- **Evaluation**: 2 modules (metrics, visualization)
- **Production**: 2 modules (inference, persistence)
- **External Integration**: 4 modules (kaggle, onnx, pmml, drift)
- **Distribution**: 3 modules (bundle-all, examples, parent)

### **Algorithm Coverage**
- **Linear Models**: 6 algorithms (Linear/Logistic Regression, Ridge, Lasso, SGD variants)
- **Tree Models**: 5 algorithms (Decision Trees, Random Forest, Gradient Boosting)
- **Clustering**: 1 algorithm (K-Means with k-means++)
- **Total**: 15+ machine learning algorithms

### **Feature Highlights**
- ✅ **Dual-Mode Visualization**: ASCII terminal + XChart GUI with automatic fallback
- ✅ **AutoML Framework**: Automated algorithm selection and hyperparameter optimization
- ✅ **Production Ready**: High-performance inference engine with caching and monitoring
- ✅ **External Integration**: Kaggle, ONNX, PMML support for enterprise workflows
- ✅ **Comprehensive Examples**: 11 complete examples including GUI demonstrations
- ✅ **Professional Logging**: Structured logging with Logback and SLF4J
- ✅ **Model Lifecycle**: Complete persistence, versioning, and drift monitoring

## 🔗 Module Dependencies

### **Dependency Hierarchy**
```
superml-core (Foundation)
├── superml-utils
├── superml-linear-models
├── superml-tree-models
├── superml-clustering
├── superml-preprocessing
├── superml-datasets
├── superml-model-selection
├── superml-pipeline
├── superml-metrics
├── superml-visualization (+ XChart)
├── superml-inference
├── superml-persistence (+ Jackson)
├── superml-autotrainer
├── superml-kaggle (+ HttpClient)
├── superml-onnx (+ ONNX Runtime)
├── superml-pmml (+ JPMML)
├── superml-drift
├── superml-bundle-all (All modules)
└── superml-examples (All modules)
```

### **External Dependencies**
- **Commons Math3**: Mathematical utilities and linear algebra
- **Commons CSV**: CSV file processing
- **Jackson**: JSON serialization for model persistence
- **Apache HttpClient**: Kaggle API integration
- **XChart**: Professional GUI visualization (optional)
- **ONNX Runtime**: Cross-platform model deployment (optional)
- **JPMML**: PMML model exchange (optional)
- **Logback/SLF4J**: Professional logging framework

## 🚀 Usage Patterns

### **Minimal Installation** (Core ML)
```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-core</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-linear-models</artifactId>
    <version>2.1.0</version>
</dependency>
```

### **Standard Installation** (Complete ML Pipeline)
```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-bundle-all</artifactId>
    <version>2.1.0</version>
</dependency>
```

### **Specialized Use Cases**
- **Kaggle Competitions**: superml-kaggle + superml-autotrainer
- **Production Inference**: superml-inference + superml-persistence
- **Data Visualization**: superml-visualization + XChart
- **Enterprise Integration**: superml-onnx + superml-pmml
- **Model Monitoring**: superml-drift + superml-metrics

## 🏗️ Architecture Benefits

### **Modularity Advantages**
1. **Selective Dependencies**: Include only needed components
2. **Reduced Bundle Size**: Lightweight applications possible
3. **Clear Separation**: Well-defined module responsibilities
4. **Easy Testing**: Module-specific test suites
5. **Independent Updates**: Module-level versioning and updates

### **Development Benefits**
1. **Parallel Development**: Teams can work on different modules
2. **Clean Interfaces**: Well-defined module contracts
3. **Dependency Management**: Clear dependency hierarchy
4. **Code Organization**: Logical grouping of related functionality
5. **Release Flexibility**: Independent module releases possible

### **Production Benefits**
1. **Performance Optimization**: Load only necessary components
2. **Memory Efficiency**: Reduced runtime footprint
3. **Deployment Flexibility**: Choose deployment components
4. **Enterprise Integration**: Standard export formats (ONNX, PMML)
5. **Monitoring**: Built-in drift detection and performance tracking

---

**SuperML Java 2.1.0** - A comprehensive, modular machine learning framework designed for education, research, and production deployment.

#### 5. **superml-preprocessing**
- **Description**: Data preprocessing utilities
- **Contains**:
  - `StandardScaler.java`
- **Dependencies**: superml-core, commons-csv
- **Features**: Data standardization and scaling

#### 6. **superml-model-selection**
- **Description**: Model selection and validation utilities
- **Contains**:
  - `CrossValidation.java`
  - `GridSearchCV.java`
  - `ParameterGrid.java`
  - `TrainTestSplit.java`
- **Dependencies**: superml-core, superml-metrics, superml-linear-models, superml-tree-models
- **Features**: Cross-validation, hyperparameter tuning

#### 7. **superml-pipeline**
- **Description**: Pipeline utilities for chaining operations
- **Contains**:
  - `Pipeline.java`
- **Dependencies**: superml-core
- **Features**: ML pipelines and workflow management

#### 8. **superml-datasets**
- **Description**: Dataset utilities and loaders
- **Contains**:
  - `DataLoaders.java`
  - `Datasets.java`
- **Dependencies**: superml-core, commons-csv, commons-io
- **Features**: Built-in datasets and data loading utilities

#### 9. **superml-inference**
- **Description**: Inference engine for production deployment
- **Contains**:
  - `BatchInferenceProcessor.java`
  - `InferenceEngine.java`
  - `OnlineInferenceService.java`
  - `PredictionService.java`
- **Dependencies**: superml-datasets, superml-persistence, superml-pipeline
- **Features**: Batch and online inference capabilities

#### 10. **superml-persistence**
- **Description**: Model persistence and serialization
- **Contains**:
  - `ModelManager.java`
  - `ModelPersistence.java`
  - `ModelRegistry.java`
- **Dependencies**: superml-core, superml-pipeline, superml-metrics, jackson-databind, commons-io
- **Features**: Model saving/loading, versioning

#### 11. **superml-metrics**
- **Description**: Evaluation metrics for ML models
- **Contains**:
  - `Metrics.java`
- **Dependencies**: superml-core
- **Features**: Classification and regression metrics

### Advanced Modules (Placeholder)

#### 12. **superml-utils**
- **Description**: General utilities
- **Status**: Empty (ready for future utilities)

#### 13. **superml-kaggle**
- **Description**: Kaggle integration utilities
- **Status**: Empty (ready for Kaggle API integration)

#### 14. **superml-autotrainer**
- **Description**: Automated machine learning capabilities
- **Status**: Empty (ready for AutoML features)

#### 15. **superml-onnx**
- **Description**: ONNX format support
- **Status**: Empty (ready for ONNX integration)

#### 16. **superml-pmml**
- **Description**: PMML format support
- **Status**: Empty (ready for PMML integration)

### Bundle and Examples

#### 17. **superml-bundle-all**
- **Description**: Complete SuperML framework with all algorithms
- **Dependencies**: All core and utility modules
- **Usage**: Single dependency to get all functionality

#### 18. **superml-examples**
- **Description**: Example code demonstrating framework usage
- **Contains**: All example files from the examples directory
- **Dependencies**: All modules for comprehensive examples

## Usage

### Including Specific Modules

To use only specific functionality, include only the modules you need:

```xml
<!-- For linear models only -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-linear-models</artifactId>
    <version>2.1.0</version>
</dependency>

<!-- For tree models only -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-tree-models</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Including Everything

To use all functionality:

```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-bundle-all</artifactId>
    <version>2.1.0</version>
</dependency>
```

## Build Instructions

### Building All Modules

```bash
mvn clean compile
```

### Building Specific Module

```bash
mvn clean compile -pl superml-linear-models
```

### Building with Dependencies

```bash
mvn clean compile -am -pl superml-linear-models
```

## Benefits of Modular Architecture

1. **Reduced Dependencies**: Users can include only what they need
2. **Better Maintainability**: Each module has a clear responsibility
3. **Easier Testing**: Modules can be tested independently
4. **Cleaner Separation**: Logical separation of concerns
5. **Future Extensibility**: Easy to add new modules without affecting existing ones
6. **Deployment Flexibility**: Different deployment scenarios can use different module combinations

## Algorithm Distribution

- **Linear Models**: 6 algorithms (Linear/Logistic Regression, SGD variants, Ridge, Lasso)
- **Tree Models**: 3 algorithms (Decision Trees, Random Forest)
- **Clustering**: 1 algorithm (K-Means)
- **Preprocessing**: 1 utility (Standard Scaler)
- **Total**: 11 implemented algorithms

## Future Extensions

The modular structure makes it easy to add:
- Neural network modules (superml-neural)
- Deep learning integration (superml-dl)
- Time series algorithms (superml-timeseries)
- Natural language processing (superml-nlp)
- Computer vision (superml-cv)
- Distributed computing (superml-spark)

## Migration Notes

The modular structure maintains backward compatibility. Existing code should work by including the `superml-bundle-all` dependency, which provides access to all classes in their original packages.
