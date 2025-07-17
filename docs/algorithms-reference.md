# Algorithms Reference Guide

SuperML Java provides a comprehensive collection of machine learning algorithms, each implemented with production-ready features and scikit-learn compatible APIs. This guide provides detailed information about all available algorithms, their capabilities, and usage examples.

## 📊 Algorithm Categories Overview

| Category | Algorithms | Use Cases |
|----------|------------|-----------|
| **Linear Models** | 6 algorithms | Classification, Regression, Feature Selection |
| **Tree-Based** | 4 algorithms | Non-linear patterns, Feature importance, Ensemble learning |
| **Neural Networks** | 3 algorithms | Deep learning, Complex patterns, Image/Sequence processing |
| **Clustering** | 1 algorithm | Unsupervised grouping, Customer segmentation |
| **Meta-Classifiers** | 1 algorithm | Multiclass conversion, Algorithm composition |
| **Preprocessing** | 1 transformer | Feature scaling, Data normalization |

## 🔢 Linear Models

### LogisticRegression
**Purpose**: Binary and multiclass classification with probabilistic outputs

**Key Features**:
- Automatic multiclass handling (One-vs-Rest and Softmax strategies)
- L1/L2 regularization support
- Gradient descent optimization with convergence monitoring
- Probability prediction capabilities
- Configurable learning rate and iterations

**Parameters**:
```java
LogisticRegression lr = new LogisticRegression()
    .setMaxIter(1000)           // Maximum iterations
    .setTol(1e-6)               // Convergence tolerance
    .setLearningRate(0.01)      // Learning rate
    .setRegularization("l2")    // Regularization type
    .setRegularizationStrength(1.0);  // Regularization strength
```

**Best Use Cases**:
- Binary classification problems
- Multiclass classification with moderate number of classes
- When probability estimates are needed
- Baseline classification model

---

### LinearRegression
**Purpose**: Ordinary least squares regression for continuous target prediction

**Key Features**:
- Closed-form solution using normal equation
- No hyperparameters to tune
- Fast training and prediction
- R² score evaluation

**Parameters**:
```java
LinearRegression lr = new LinearRegression()
    .setFitIntercept(true);     // Whether to fit intercept term
```

**Best Use Cases**:
- Linear relationships between features and target
- Baseline regression model
- When interpretability is important
- Small to medium datasets

---

### Ridge
**Purpose**: L2 regularized regression to prevent overfitting

**Key Features**:
- L2 regularization (weight decay)
- Closed-form solution with regularization
- Handles multicollinearity well
- Cross-validation compatible

**Parameters**:
```java
Ridge ridge = new Ridge()
    .setAlpha(1.0)              // Regularization strength
    .setFitIntercept(true);     // Whether to fit intercept
```

**Best Use Cases**:
- High-dimensional datasets
- When features are correlated
- Preventing overfitting in linear models
- When all features should be retained

---

### Lasso
**Purpose**: L1 regularized regression with automatic feature selection

**Key Features**:
- L1 regularization for feature selection
- Coordinate descent optimization
- Sparse solutions (some coefficients become zero)
- Built-in feature selection

**Parameters**:
```java
Lasso lasso = new Lasso()
    .setAlpha(1.0)              // Regularization strength
    .setMaxIter(1000)           // Maximum iterations
    .setTol(1e-4);              // Convergence tolerance
```

**Best Use Cases**:
- Feature selection in high-dimensional data
- When interpretability is crucial
- Sparse data or when many features are irrelevant
- Automatic model simplification

---

### SoftmaxRegression
**Purpose**: Direct multinomial classification with softmax activation

**Key Features**:
- Native multiclass support (no meta-learning needed)
- Softmax activation for probability normalization
- Cross-entropy loss optimization
- Gradient descent with momentum

**Parameters**:
```java
SoftmaxRegression softmax = new SoftmaxRegression()
    .setMaxIter(1000)           // Maximum iterations
    .setLearningRate(0.01)      // Learning rate
    .setTol(1e-6);              // Convergence tolerance
```

**Best Use Cases**:
- Multiclass classification with many classes
- When class probabilities are needed
- Text classification
- Image classification

---

### OneVsRestClassifier
**Purpose**: Meta-classifier that converts any binary classifier into multiclass

**Key Features**:
- Works with any binary classifier
- Trains one classifier per class
- Probability calibration and normalization
- Parallel training support

**Parameters**:
```java
OneVsRestClassifier ovr = new OneVsRestClassifier(new LogisticRegression())
    .setNJobs(4);               // Number of parallel jobs
```

**Best Use Cases**:
- Converting binary algorithms to multiclass
- When you want to use specific binary algorithms for multiclass
- Large number of classes
- When different classes have different characteristics

## 🌳 Tree-Based Models

### DecisionTree
**Purpose**: Non-linear classification and regression using tree-based decisions

**Key Features**:
- CART (Classification and Regression Trees) implementation
- Multiple splitting criteria: Gini, Entropy, MSE
- Comprehensive pruning controls
- Handles both numerical and categorical features
- Feature importance calculation

**Parameters**:
```java
DecisionTree dt = new DecisionTree()
    .setCriterion("gini")           // Splitting criterion
    .setMaxDepth(10)                // Maximum tree depth
    .setMinSamplesSplit(2)          // Min samples to split node
    .setMinSamplesLeaf(1)           // Min samples in leaf
    .setMinImpurityDecrease(0.0)    // Min impurity decrease for split
    .setMaxFeatures(-1)             // Max features to consider (-1 = all)
    .setRandomState(42);            // Random seed
```

**Best Use Cases**:
- Non-linear relationships
- Mixed data types (numerical + categorical)
- When interpretability is important
- Feature selection and importance analysis
- Baseline for ensemble methods

---

### RandomForest
**Purpose**: Ensemble of decision trees with bootstrap aggregating

**Key Features**:
- Bootstrap sampling for each tree
- Random feature selection at each split
- Parallel training capabilities
- Out-of-bag (OOB) error estimation
- Feature importance aggregation
- Robust to overfitting

**Parameters**:
```java
RandomForest rf = new RandomForest()
    .setNEstimators(100)            // Number of trees
    .setMaxDepth(10)                // Maximum depth per tree
    .setCriterion("gini")           // Splitting criterion
    .setMinSamplesSplit(2)          // Min samples to split
    .setMinSamplesLeaf(1)           // Min samples in leaf
    .setMaxFeatures("sqrt")         // Features per split
    .setBootstrap(true)             // Bootstrap sampling
    .setOobScore(true)              // Calculate OOB score
    .setNJobs(-1)                   // Parallel jobs (-1 = all cores)
    .setRandomState(42);            // Random seed
```

**Best Use Cases**:
- General-purpose classification and regression
- When high accuracy is needed
- Large datasets with many features
- When overfitting is a concern
- Feature importance analysis

---

### GradientBoosting
**Purpose**: Sequential ensemble that builds trees to correct previous errors

**Key Features**:
- Sequential learning with gradient descent
- Early stopping with validation monitoring
- Subsampling (stochastic gradient boosting)
- Configurable learning rate and regularization
- Training and validation score tracking
- Feature importance calculation

**Parameters**:
```java
GradientBoosting gb = new GradientBoosting()
    .setNEstimators(100)            // Number of boosting stages
    .setLearningRate(0.1)           // Shrinkage parameter
    .setMaxDepth(3)                 // Maximum depth per tree
    .setSubsample(1.0)              // Fraction of samples per tree
    .setMinSamplesSplit(2)          // Min samples to split
    .setMinSamplesLeaf(1)           // Min samples in leaf
    .setMinImpurityDecrease(0.0)    // Min impurity decrease
    .setValidationFraction(0.1)     // Validation set fraction
    .setNIterNoChange(5)            // Early stopping patience
    .setTol(1e-4)                   // Early stopping tolerance
    .setRandomState(42);            // Random seed
```

**Best Use Cases**:
- High-accuracy classification and regression
- Competitions and benchmarks
- When careful tuning can be done
- Complex non-linear relationships
- When overfitting can be controlled

## 🧠 Neural Networks

### MLPClassifier
**Purpose**: Multi-Layer Perceptron for classification with deep learning capabilities

**Key Features**:
- Configurable hidden layer architecture
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Batch processing and mini-batch training
- Early stopping and validation monitoring
- Gradient descent optimization with momentum

**Parameters**:
```java
MLPClassifier mlp = new MLPClassifier()
    .setHiddenLayerSizes(128, 64, 32)    // Hidden layer architecture
    .setActivation("relu")               // Activation function
    .setLearningRate(0.001)              // Learning rate
    .setMaxIter(200)                     // Maximum epochs
    .setBatchSize(32)                    // Batch size
    .setEarlyStopping(true)              // Enable early stopping
    .setValidationFraction(0.2);         // Validation split
```

**Best Use Cases**:
- Complex classification problems with non-linear patterns
- High-dimensional data
- When sufficient training data is available
- Feature interaction modeling

---

### CNNClassifier
**Purpose**: Convolutional Neural Network for image classification and spatial pattern recognition

**Key Features**:
- Convolutional and pooling layers
- Automatic feature extraction from images
- Configurable CNN architecture
- Batch normalization and dropout support
- GPU acceleration ready

**Parameters**:
```java
CNNClassifier cnn = new CNNClassifier()
    .setInputShape(32, 32, 3)            // Image dimensions (H, W, C)
    .setConvLayers(32, 64, 128)          // Convolutional filters
    .setKernelSize(3)                    // Convolution kernel size
    .setPoolingSize(2)                   // Max pooling size
    .setDropoutRate(0.3)                 // Dropout for regularization
    .setLearningRate(0.001);             // Learning rate
```

**Best Use Cases**:
- Image classification tasks
- Computer vision problems
- Spatial pattern recognition
- When translation invariance is important

---

### RNNClassifier
**Purpose**: Recurrent Neural Network for sequence classification and temporal pattern recognition

**Key Features**:
- LSTM and GRU cell support
- Variable sequence length handling
- Bidirectional processing
- Attention mechanisms
- Memory state management

**Parameters**:
```java
RNNClassifier rnn = new RNNClassifier()
    .setSequenceLength(100)              // Maximum sequence length
    .setHiddenSize(128)                  // Hidden state size
    .setNumLayers(2)                     // Number of RNN layers
    .setCellType("LSTM")                 // Cell type (LSTM/GRU)
    .setBidirectional(true)              // Bidirectional processing
    .setLearningRate(0.001);             // Learning rate
```

**Best Use Cases**:
- Sequence classification (text, time series)
- Natural language processing
- Time series prediction
- When temporal dependencies matter

## 🎯 Clustering

### KMeans
**Purpose**: Partitioning clustering for grouping similar data points

**Key Features**:
- K-means++ initialization for better convergence
- Multiple random restarts to avoid local minima
- Inertia (within-cluster sum of squares) calculation
- Configurable convergence criteria
- Cluster center and label prediction

**Parameters**:
```java
KMeans kmeans = new KMeans()
    .setNClusters(8)                // Number of clusters
    .setInit("k-means++")           // Initialization method
    .setNInit(10)                   // Number of initializations
    .setMaxIter(300)                // Maximum iterations
    .setTol(1e-4)                   // Convergence tolerance
    .setRandomState(42);            // Random seed
```

**Best Use Cases**:
- Customer segmentation
- Market research
- Image segmentation
- Data exploration and visualization
- Dimensionality reduction preprocessing

## 🔧 Preprocessing

### StandardScaler
**Purpose**: Feature standardization to zero mean and unit variance

**Key Features**:
- Z-score normalization (mean=0, std=1)
- Fit/transform pattern consistent with scikit-learn
- Feature-wise scaling independence
- Inverse transformation capability
- Numerical stability

**Parameters**:
```java
StandardScaler scaler = new StandardScaler()
    .setWithMean(true)              // Center to zero mean
    .setWithStd(true);              // Scale to unit variance
```

**Best Use Cases**:
- Preprocessing for linear models
- When features have different scales
- Before clustering algorithms
- Neural network preprocessing
- SVM preprocessing

## 📈 Performance Characteristics

### Training Time Complexity

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| **LogisticRegression** | O(n × p × i) | O(p) | n=samples, p=features, i=iterations |
| **LinearRegression** | O(n × p²) | O(p²) | Matrix inversion |
| **Ridge** | O(n × p²) | O(p²) | Matrix inversion with regularization |
| **Lasso** | O(n × p × i) | O(p) | Coordinate descent iterations |
| **DecisionTree** | O(n × p × log n) | O(n) | Average case for balanced tree |
| **RandomForest** | O(t × n × p × log n) | O(t × n) | t=number of trees |
| **GradientBoosting** | O(b × n × p × log n) | O(b × n) | b=boosting iterations |
| **KMeans** | O(n × k × i × p) | O(n × p) | k=clusters, i=iterations |

### Prediction Time Complexity

| Algorithm | Time Complexity | Notes |
|-----------|----------------|-------|
| **Linear Models** | O(p) | Simple linear combination |
| **DecisionTree** | O(log n) | Tree traversal depth |
| **RandomForest** | O(t × log n) | t trees × tree depth |
| **GradientBoosting** | O(b × log n) | b boosting stages × tree depth |
| **KMeans** | O(k × p) | Distance to k centroids |

## 🎯 Algorithm Selection Guide

### For Classification Problems

| Problem Type | Recommended Algorithm | Alternative |
|--------------|----------------------|-------------|
| **Linear separable** | LogisticRegression | SoftmaxRegression |
| **Non-linear** | RandomForest | GradientBoosting |
| **High dimensions** | LogisticRegression + L1 | Lasso |
| **Many classes** | SoftmaxRegression | OneVsRestClassifier |
| **Interpretability** | DecisionTree | LogisticRegression |
| **High accuracy** | GradientBoosting | RandomForest |

### For Regression Problems

| Problem Type | Recommended Algorithm | Alternative |
|--------------|----------------------|-------------|
| **Linear relationship** | LinearRegression | Ridge |
| **Feature selection** | Lasso | Ridge + manual selection |
| **Non-linear** | RandomForest | GradientBoosting |
| **Multicollinearity** | Ridge | LinearRegression |
| **High accuracy** | GradientBoosting | RandomForest |
| **Interpretability** | DecisionTree | LinearRegression |

### For Clustering Problems

| Problem Type | Recommended Algorithm | Notes |
|--------------|----------------------|-------|
| **Spherical clusters** | KMeans | Works best with globular clusters |
| **Unknown cluster count** | KMeans + Elbow method | Try different k values |

## 🚀 Advanced Features

### Ensemble Capabilities
- **RandomForest**: Bootstrap aggregating with feature randomization
- **GradientBoosting**: Sequential boosting with early stopping
- **OneVsRestClassifier**: Meta-learning for multiclass conversion

### Regularization Support
- **L1 Regularization**: Lasso, LogisticRegression
- **L2 Regularization**: Ridge, LogisticRegression
- **Early Stopping**: GradientBoosting with validation monitoring

### Parallel Processing
- **RandomForest**: Multi-threaded tree training
- **OneVsRestClassifier**: Parallel binary classifier training
- **GridSearchCV**: Parallel hyperparameter optimization

### Probability Estimation
- **LogisticRegression**: Native probability support
- **SoftmaxRegression**: Multinomial probabilities
- **RandomForest**: Voting-based probabilities
- **GradientBoosting**: Sigmoid-transformed probabilities

## 📚 Usage Examples

### Complete Classification Workflow
```java
// Load and prepare data
var dataset = Datasets.makeClassification(1000, 20, 3);
var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);

// Preprocessing
StandardScaler scaler = new StandardScaler();
double[][] XTrainScaled = scaler.fitTransform(split.XTrain);
double[][] XTestScaled = scaler.transform(split.XTest);

// Train multiple algorithms
LogisticRegression lr = new LogisticRegression().fit(XTrainScaled, split.yTrain);
RandomForest rf = new RandomForest(100, 10).fit(XTrainScaled, split.yTrain);
GradientBoosting gb = new GradientBoosting(100, 0.1, 6).fit(XTrainScaled, split.yTrain);

// Evaluate and compare
double lrAccuracy = Metrics.accuracy(split.yTest, lr.predict(XTestScaled));
double rfAccuracy = Metrics.accuracy(split.yTest, rf.predict(XTestScaled));
double gbAccuracy = Metrics.accuracy(split.yTest, gb.predict(XTestScaled));

System.out.printf("Logistic Regression: %.3f\n", lrAccuracy);
System.out.printf("Random Forest: %.3f\n", rfAccuracy);
System.out.printf("Gradient Boosting: %.3f\n", gbAccuracy);
```

### Hyperparameter Optimization
```java
// Grid search for Random Forest
Map<String, Object[]> paramGrid = Map.of(
    "n_estimators", new Object[]{50, 100, 200},
    "max_depth", new Object[]{5, 10, 15},
    "min_samples_split", new Object[]{2, 5, 10}
);

GridSearchCV gridSearch = new GridSearchCV(new RandomForest(), paramGrid)
    .setCv(5)
    .setScoring("accuracy")
    .setNJobs(-1);

gridSearch.fit(XTrainScaled, split.yTrain);
RandomForest bestRF = (RandomForest) gridSearch.getBestEstimator();
```

This comprehensive algorithms reference provides detailed information about all available algorithms in SuperML Java, their capabilities, parameters, and best use cases. Each algorithm is implemented with production-ready features and follows scikit-learn compatible APIs for easy adoption.
