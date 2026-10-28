# MachinePULSE
### ANN-Based Machine Failure Risk Prediction

**Artificial Neural Networks Lab Project | Python | NumPy | Streamlit**

MachinePULSE is an interactive machine learning application that analyzes machine sensor data and estimates the probability of machine failure. It uses a custom Artificial Neural Network (ANN) implemented from scratch with NumPy, along with data preprocessing, model evaluation, risk categorization, and feature-influence analysis.

The project demonstrates how neural networks can be developed without a high-level deep learning framework and integrated into a practical application.

---

## Table of Contents

- Project Overview
- Problem Statement
- Objectives
- Technologies Used
- ANN Architecture
- Model Training Methodology
- Risk Classification
- Evaluation Metrics
- Feature Influence Analysis
- Project Structure
- Error Handling
- Installation and Execution
- Testing
- Limitations
- Future Improvements

## 1. Project Overview

MachinePULSE processes machine sensor readings and uses a trained ANN to estimate the probability of a machine failure.

The application is designed to support sensor-data analysis through a Streamlit interface. Its processing pipeline separates data preparation, neural-network operations, model persistence, prediction, and evaluation.

The model produces a failure probability, which the application can use to categorize the estimated risk as Healthy, Warning, or Critical.

**Important:** These categories represent model-based estimates, not guaranteed machine conditions or a replacement for industrial safety systems.

## 2. Problem Statement

Industrial equipment generates sensor readings such as temperature, rotational speed, torque, and tool wear. Patterns in these readings may be associated with machine failures.

Analyzing large datasets manually can be difficult. Machine learning provides a way to learn patterns from historical sensor data and estimate failure risk for new observations.

MachinePULSE addresses this problem using a binary-classification neural network and an interactive data-processing workflow.

## 3. Project Objectives

- Implement an Artificial Neural Network from scratch using NumPy.
- Process machine sensor data using a reusable Python pipeline.
- Prepare input features and standardize numerical data.
- Address class imbalance using weighted binary cross-entropy.
- Train the model using mini-batch gradient descent with Adam optimization.
- Reduce overfitting using L2 regularization and early stopping.
- Evaluate predictions using multiple classification metrics.
- Save and reload model parameters and preprocessing information.
- Estimate individual feature influence through controlled input comparisons.
- Provide an interactive Streamlit application for data analysis.

## 4. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Neural-network calculations, matrix operations, activations, gradients, and parameter updates |
| Pandas | Tabular sensor data and DataFrame processing |
| Scikit-learn | Data splitting, feature scaling, and classification evaluation metrics |
| Streamlit | Interactive web application |
| Pickle | Saving and loading the trained model bundle |
| VS Code | Development environment |
| Git and GitHub | Source-code management and project sharing |

The neural-network architecture and its forward propagation, backpropagation, and Adam update operations are implemented directly in `ann_engine.py`.

## 5. ANN Architecture

The network follows this architecture:

**Input Layer → Dense Hidden Layers with ReLU → Dense Output Layer with Sigmoid**

The hidden-layer configuration is provided through the `hidden_layers` argument during training. For example, a configuration of `[32, 16]` creates two hidden layers containing 32 and 16 neurons, respectively.

The complete architecture is:

`Input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)`

This example illustrates one possible configuration; the actual architecture depends on the selected hidden layers.

### Input Layer

The input layer receives the numerical features prepared by the data-processing pipeline. The input dimension is determined by the number of selected features.

### Hidden Layers — ReLU

The hidden layers use the Rectified Linear Unit activation function:

`ReLU(z) = max(0, z)`

ReLU introduces non-linearity, allowing the network to learn more complex patterns in sensor data.

### Output Layer — Sigmoid

The network has one output neuron with a sigmoid activation function:

`Sigmoid(z) = 1 / (1 + exp(-z))`

The output represents an estimated probability between 0 and 1 for the positive class, which is encoded as failure (`1`).

### Weight Initialization

The network uses He initialization for its weight matrices. This initialization is designed to work well with ReLU hidden layers.

Biases are initialized to zero.

### Numerical Stability

The sigmoid implementation clips its input to the range `[-60, 60]` before calculating the exponential. This helps prevent numerical overflow.

## 6. Model Training Methodology

### Step 1: Feature Preparation

The `add_features()` function prepares the dataset for modeling. The pipeline selects the configured `FEATURES` and the binary target column named `failure`.

The precise feature-engineering transformations are implemented in `data_utils.py`.

### Step 2: Training, Validation, and Testing

The dataset is divided using stratified splitting with `random_state=42`.

- **Test set:** 20% of the original dataset.
- **Training and validation pool:** 80% of the original dataset.
- **Validation set:** 15% of the 80% training-and-validation pool.
- **Final training set:** The remaining 85% of that pool.

This means the approximate overall proportions are 68% training, 12% validation, and 20% testing.

Stratification helps preserve the class proportions across the splits.

### Step 3: Feature Standardization

`StandardScaler` standardizes the input features.

The scaler is fitted only on the training data. The same training-derived mean and scale are then used to transform the training, validation, and test data.

This prevents information from the validation and test sets from being used to fit the scaler.

### Step 4: Forward Propagation

Each layer calculates a weighted sum of its inputs and biases:

`z = aW + b`

ReLU is applied to the hidden layers, while sigmoid is applied to the final output layer.

### Step 5: Loss Function

The model uses weighted binary cross-entropy.

The positive class receives a configurable `pos_weight`, allowing failure examples to contribute more strongly to the loss when failures are relatively rare.

The default value in `NeuralNetwork.fit()` is `5.0`, although the training pipeline passes its configured value to the network.

### Step 6: Backpropagation

The network calculates gradients for weights and biases by propagating the output error backward through the layers.

The implementation uses the sigmoid-plus-cross-entropy output error term and the derivative of ReLU for hidden layers.

### Step 7: Adam Optimization

The network updates its parameters using the Adam optimizer.

The implementation uses the following standard Adam settings:

- Learning-rate default: `0.003`
- First-moment coefficient: `0.9`
- Second-moment coefficient: `0.999`
- Numerical stability constant: `1e-8`

The learning rate can be configured during model training.

### Step 8: Regularization and Early Stopping

L2 regularization adds a weight penalty to the weight gradients, helping discourage excessively large weights.

Early stopping monitors validation loss. The implementation keeps the best-performing weights observed on the validation data and stops training when the configured patience is exhausted.

The default maximum number of epochs is 80, the default batch size is 64, and the default patience is 20. The training pipeline can supply different settings.

## 7. Risk Classification

The pipeline converts the model's estimated failure probability into a risk category using two thresholds.

| Estimated probability | Risk category |
|---|---|
| Below 0.25 | Healthy |
| 0.25 to below 0.50 | Warning |
| 0.50 or above | Critical |

The thresholds are defined in `pipeline.py`:

- `WARNING_LEVEL = 0.25`
- `CRITICAL_LEVEL = 0.50`

These thresholds control risk categorization. They do not guarantee the actual physical condition of a machine.

## 8. Model Evaluation

The model evaluates predictions on the held-out test set.

The following metrics are calculated:

### Accuracy

The proportion of test observations classified correctly.

### Precision

The proportion of predicted failures that are actual failures.

### Recall

The proportion of actual failures correctly identified by the model.

### F1-Score

The harmonic mean of precision and recall, balancing both metrics.

### ROC-AUC

Measures how well the model separates positive and negative examples across different probability thresholds.

### Confusion Matrix

Summarizes correct and incorrect predictions for each class.

The training pipeline also records training loss and validation loss across epochs, the number of test observations, and the number of trainable parameters.

**Actual evaluation results should be reported from a completed training run.** No accuracy, recall, F1-score, or ROC-AUC value is assumed in this documentation.

## 9. Feature Influence Analysis

The `feature_influence()` function provides a simple feature-influence estimate.

It first calculates the predicted failure probability for an observation. It then changes one standardized feature at a time to zero while keeping the other features fixed.

Because standardized zero corresponds to the training mean, the change in prediction estimates how the selected feature affects that observation's model output relative to its average value.

This method is a model-sensitivity heuristic. It is not a causal explanation and should not be described as SHAP or as a guarantee of feature importance.

## 10. Model Saving and Loading

The pipeline saves a model bundle to:

`models/machinepulse_model.pkl`

The bundle contains:

- Neural-network weights and biases.
- Layer configuration.
- Feature names.
- Standardization means and scales.
- Selected hidden-layer configuration.
- Trainable parameter count.
- Evaluation metrics.
- Confusion matrix.
- Training and validation loss history.
- Test-set size.
- A flag indicating whether the model was trained on demonstration data.

Pickle is used for model persistence. Only load pickle files from trusted sources because deserializing untrusted pickle files can execute malicious code.

## 11. Project Structure

```text
MachinePULSE/
├── app.py
├── ann_engine.py
├── data_utils.py
├── pipeline.py
├── requirements.txt
├── data/
├── models/
├── .streamlit/
│   └── config.toml
├── screenshots/
└── README.md
```

### `app.py`

Provides the Streamlit user interface and connects user interactions with the data-processing and prediction workflow.

### `ann_engine.py`

Implements the neural network from scratch, including:

- ReLU and sigmoid activations.
- Weight initialization.
- Forward propagation.
- Backpropagation.
- Adam parameter updates.
- Weighted binary cross-entropy.
- L2 regularization.
- Mini-batch training.
- Early stopping.
- Probability and class predictions.
- Parameter counting and model serialization.

### `data_utils.py`

Defines the base project directory, selected features, and feature-engineering functionality.

### `pipeline.py`

Connects data preparation, splitting, scaling, training, evaluation, model saving/loading, prediction, and feature-influence analysis.

### `requirements.txt`

Lists the Python dependencies needed to install and run the application.

### `data/`

Contains the dataset or sample sensor-data files used by the project.

### `models/`

Stores the trained model bundle and other required model artifacts.

## 12. Installation and Execution

### Requirements

- Python
- pip
- Project source files
- Required dataset and model artifacts, where applicable

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/MachinePULSE.git
cd MachinePULSE
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 13. Error Handling and Reliability

The application workflow should account for common data and environment issues.

- **Invalid CSV:** A file that cannot be parsed should produce a readable error message.
- **Missing columns:** Input data must include the columns required by the feature-processing pipeline.
- **Invalid numerical values:** Sensor features must be converted to valid numerical inputs before model inference.
- **Missing model file:** The application should guide the user to train or load a model before prediction.
- **Dependency errors:** Required libraries must be installed in the active Python environment.
- **Numerical overflow:** Clipping the sigmoid input helps maintain numerical stability.

The actual exception-handling behavior depends on the implementation in the application and utility modules.

## 14. Testing Plan

The following tests can be used to verify the application:

1. Launch the application successfully.
2. Load a valid sample sensor CSV.
3. Upload a correctly formatted CSV.
4. Check the handling of missing required columns.
5. Check the handling of unreadable files.
6. Train the model and verify that the model bundle is saved.
7. Reload the saved model bundle.
8. Verify that prediction probabilities fall between zero and one.
9. Confirm that the risk thresholds produce the expected categories.
10. Compare the displayed evaluation metrics with the values calculated by the pipeline.

Screenshots and actual test outcomes should be added after these tests are completed.

## 15. Limitations

- Model quality depends on the dataset, feature engineering, and training configuration.
- Rare failure events can make accuracy alone misleading.
- The selected probability thresholds may require calibration for a particular application.
- Feature influence estimates describe model behavior, not causation.
- Predictions on unseen machine types or out-of-distribution data may be unreliable.
- The application is an academic demonstration and is not a certified industrial safety or predictive-maintenance system.

## 16. Future Improvements

- Add precision-recall curves and ROC curves.
- Compare multiple ANN architectures.
- Calibrate predicted probabilities and risk thresholds.
- Improve data validation and missing-value handling.
- Add downloadable prediction reports.
- Add model-version tracking and reproducible experiment records.
- Deploy the Streamlit application to a public hosting platform.
- Add monitoring and feedback from real machine observations.

## 17. Conclusion

MachinePULSE demonstrates a complete machine learning workflow combining a NumPy-based Artificial Neural Network, feature engineering, standardization, class-imbalance handling, Adam optimization, regularization, early stopping, model evaluation, and an interactive Streamlit application.

The project provides practical experience in understanding neural-network internals and integrating a custom model into a data-driven application.

## Academic Information

- **Project Name:** MachinePULSE
- **Course:** Artificial Neural Networks Lab
- **Primary Language:** Python
- **Neural-Network Implementation:** NumPy
- **Preprocessing and Evaluation:** Scikit-learn
- **User Interface:** Streamlit
- **Development Environment:** Visual Studio Code

## License

Choose and add an appropriate license before permitting others to reuse or redistribute the project.
