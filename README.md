# README: AI-Powered Stock Market Forecasting and Risk Analysis for the NZX 50

## Project Title

**AI-Powered Stock Market Forecasting and Risk Analysis for the NZX 50: An Ethical Ensemble Learning Approach**

## Author

**Ray Chakanetsa Marange**
School of Engineering and Computer Science
Victoria University of Wellington
Wellington, New Zealand
Email: ray.marange@ecs.vuw.ac.nz — Student ID: 300671111

## Overview

This project implements a comprehensive **Financial Forecasting and Risk Analysis Pipeline** for the **NZX 50 index**. The system emphasizes model **robustness** during market stress, **interpretability** via Explainable AI (**XAI**), and **ethical alignment** using **Tikanga Māori** principles. The core approach utilizes an **Ensemble Learning** strategy to generate highly calibrated and confidence-boosted forecasts.

## 1\. System Architecture and Data Flow

Based on the provided Python code files, particularly `main.ipynb`, `data_pipeline.ipynb`, `evaluator.ipynb`, `xai_analyzer.ipynb`, and `ensemble_model.ipynb`, the system is structured as a comprehensive Financial Forecasting and Risk Analysis Pipeline for the NZX 50 index.

### 1.1. Data Flow Pipeline for Reporting

This pipeline illustrates the sequence of data transformation, training, evaluation, and reporting steps from raw market data to the final saved results and visualizations.

| Phase | Component/Function | Input Data | Output Data | Key Operations |
| :---: | :---: | :---: | :---: | :--- |
| **P0: Setup** | `main.ipynb` (setup\_device, setup\_experiment\_dirs, RequirementsConfig.validate\_installation) | Configuration Files (`config.ipynb`) | Device (`cuda`/`cpu`), Log/Model/XAI Directories | Initialization, Dependency Check |
| **P1: Data Prep** | `FinancialDataPipeline` (`data_pipeline.ipynb`) | Raw Market Tickers (`^NZ50`, `SPY`, etc.) | `forecast_data` | Fetching (yfinance), Feature Engineering (RSI, MACD, Volatility, Liquidity), Regime Detection (Stress/Normal), Time-Series Splitting |
| **P2: Model Setup** | `ModelConfig.create_forecasting_model_configs`, `ModelFactory.create_model`, `create_forecast_loaders` | `forecast_data`, `TrainingConfig` | `models` (PyTorch instances: Transformer, LSTM, Linear; XGBoost), `data_loaders` (Train, Val, Test) | Model Initialization, Hyperparameter Configuration, Data Loading and Batching |
| **P3: Training** | `main.ipynb` (train\_enhanced\_forecasting\_models) | `models`, `data_loaders` | `trainers` (Trained Model Instances) | Regime-Aware Loss Weighting, Gradient Accumulation, Gradient Clipping, Early Stopping, XGBoost Fitting |
| **P4: Evaluation** | `UnifiedModelEvaluator` (`evaluator.ipynb`) | `trainers`, `data_loaders['test']` | `evaluation_results` | Direction Accuracy, Sharpe Ratio, Prediction Correlation, Performance Gap during Stress Periods |
| **P5: XAI Analysis** | `perform_enhanced_xai_analysis` (`symbolic_xai.ipynb`, `xai_analyzer.ipynb`) | `evaluation_results`, `data_loaders['test']` | `xai_results` | Transparency/Fairness Score, Tikanga Māori Assessment, Symbolic Regression Equation (RQ6), Counterfactuals |
| **P6: Comparative Analysis** | `perform_comparative_analysis`, `address_enhanced_research_questions` | `evaluation_results`, `xai_results`, `coverage_info` | Console Output (RQ Analysis Summary, Metrics Table) | Comparison of Accuracy vs. Interpretability, Regulatory Alignment Check, Stress Performance Ranking |
| **P7: Forecasting** | `generate_confidence_boosted_forecast` (`ensemble_model.ipynb`) | Current Price (Live/Historical), Historical Features, `ensemble_models` | `forecast_results` (6M/1M Return/Price, Calibrated Confidence, Risk Factors) | Ensemble Weighting, Confidence Calibration (Agreement, Volatility), Risk Assessment |
| **P8: Reporting & Output** | `save_enhanced_final_results`, `generate_xai_visualizations` | All Analysis Results | `enhanced_final_results.json`, `enhanced_forecasting_comparison.csv`, XAI Visualizations (`.png`) | Serialization, Plotting (Radar Charts, Feature Importance), Final Execution Summary |

### 1.2. System Architecture / System Overview

This diagram outlines the core components, modules, and their interactions, highlighting the pipeline's emphasis on e**X**plainable **AI** (XAI) and performance robustness.

| Component | Modules/Classes | Key Functionality |
| :--- | :--- | :--- |
| **Data Layer** | `DataConfig` (`config.ipynb`), `FinancialDataPipeline` (`data_pipeline.ipynb`) | Fetches/Cleans NZX 50 and global data, engineers features (volatility, illiquidity, regime), creates balanced, leakage-free time-series sequences. |
| **Model Layer** | `AdvancedModelFactory`, `EnhancedMultiHorizonTransformer` (`model_architectures.ipynb`) | Instantiates sequence models (Transformer/LSTM) for multi-horizon forecasting, and tree-based/linear baselines. Handles model initialization and complexity tracking. |
| **Training Layer** | `AdvancedRegimeAwareTrainer` (`evaluator.ipynb`) | Handles model training with a focus on stability and robustness: **adaptive stress-period weighting**, gradient accumulation, learning rate scheduling. |
| **Evaluation Layer** | `UnifiedModelEvaluator` (`evaluator.ipynb`) | Comprehensive performance assessment: MSE, Sharpe Ratio, **Direction Accuracy**, and dedicated **Enhanced Stress Analysis** to measure robustness. |
| **XAI & Interpretability Layer** | `EnhancedXAIAnalysis`, `SymbolicMarketRegressor` (`symbolic_xai.ipynb`), `XAIAnalyzer` (`xai_analyzer.ipynb`) | Generates human-readable explanations: **Symbolic Regression** (RQ6) to find mathematical rules, **Ethical Assessment (Tikanga Māori principles)**, SHAP analysis, and counterfactuals. |
| **Forecasting & Ensemble Layer** | `generate_confidence_boosted_forecast` (`ensemble_model.ipynb`) | Generates the final 6-month and 1-month forecasts, prioritizes a **Confidence-Boosted Ensemble** prediction, and performs real-time volatility-based risk assessment. |

-----

## 2\. Setup and Dependencies

### 2.1. Prerequisites

You must have **Python 3.8+** installed. This project is optimized for running within **Jupyter** or **Google Colab** environments and requires a machine with **NVIDIA CUDA support** for GPU acceleration.

### 2.2. Installation

All dependencies are listed in the `requirements.txt` file.

1.  **Install Base Environment:**

    ```bash
    # Install all required Python packages (including PyTorch CPU version)
    pip install -r requirements.txt
    # Also ensure Jupyter is installed (if not using Anaconda)
    pip install jupyter notebook
    ```

2.  **GPU (CUDA) Configuration (Crucial for Performance):**
    To leverage your NVIDIA GPU, you must install the **CUDA Toolkit** and the **CUDA-enabled version of PyTorch**.

      * **Prerequisite:** Ensure you have the NVIDIA drivers and CUDA Toolkit installed on your system (check NVIDIA's official documentation for this step).
      * **Install CUDA PyTorch:** You will need to install PyTorch compiled for your specific CUDA version (e.g., CUDA 12.1).
        ```bash
        # Example for PyTorch with a supported CUDA version (Check pytorch.org for the exact command)
        # This command is often required for GPU acceleration:
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXX
        ```

### 2.3. Dependencies List

| Category | Package | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Core ML/DL** | `torch`, `pandas`, `numpy`, `sklearn`, `scipy` | `>=2.0.0`, `>=1.5.0`, `>=1.21.0`, `>=1.2.0`, `>=1.9.0` | Data manipulation, matrix operations, core deep learning, statistical analysis. |
| **Data Acquisition** | `yfinance` | `>=0.2.0` | Fetches historical and live financial data. |
| **XAI/Interpretability** | `gplearn`, `shap`, `sympy`, `interpret` | `==0.4.2`, `>=0.42.0`, `>=1.10`, `>=0.4.0` | **Symbolic Regression** (RQ6), SHAP analysis, and symbolic mathematics. |
| **Utilities/Viz** | `matplotlib`, `seaborn`, `plotly`, `tqdm`, `ipywidgets` | `>=3.5.0`, `>=0.12.0`, `>=5.10.0`, `>=4.64.0`, `>=8.0.0` | Plotting, interactive visualizations, and progress bars. |

-----

## 3\. How to Run (Local and Cloud)

### 3.1. Jupyter Notebook Execution (Interactive - Recommended Local)

This is the preferred method for local execution, allowing interactive debugging, analysis of results, and inline visualization.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Open `main.ipynb`:** Navigate to and open the `main.ipynb` file in the Jupyter dashboard.
3.  **Run the Pipeline:** The entire pipeline can be executed by selecting **Cell \> Run All** or by using the `%run` magic command in the first cell of a separate notebook.

### 3.2. **Google Colab Execution (Cloud GPU - Recommended)**

Google Colaboratory provides a free, pre-configured environment with NVIDIA GPUs (T4, V100, etc.) which is ideal for this project.

1.  **Open Colab:** Go to Google Colab and create a new notebook.
2.  **Set Runtime:** Go to **Runtime \> Change runtime type** and select **GPU** as the hardware accelerator.
3.  **Clone Repository:** Execute the following shell commands in the first cells (replace `<repo_url>` with your actual repository URL):
    ```bash
    !git clone <repo_url>
    %cd <project_directory_name>
    ```
4.  **Install Dependencies:** Install the required external dependencies.
    ```bash
    # The -q flag minimizes installation output
    !pip install -r requirements.txt -q
    ```
5.  **Run Pipeline:** Execute the main script using the `%run` magic command (or open the `main.ipynb` notebook and run all cells).
    ```python
    %run main.ipynb
    ```
    The `main.ipynb` notebook's logic will automatically detect the Colab GPU and use it for training.

### 3.3. **GPU (CUDA) Verification and Execution**

The application is configured to automatically detect and use CUDA (or MPS on Apple Silicon) when available.

1.  **Verify CUDA Setup (Any Notebook):** Confirm PyTorch can access your GPU by running this in any notebook cell:
    ```python
    import torch
    print('CUDA available:', torch.cuda.is_available())
    # Should print 'CUDA available: True'
    ```
2.  **Execution with GPU:** Running the notebook will use the GPU, as the core logic is designed to prioritize it.

-----

## 4\. Project File Structure

```
.
├── main.ipynb                   # Main execution script and pipeline orchestration
├── config.ipynb                 # Configuration settings (data, training, XAI, models)
├── data_pipeline.ipynb          # Data fetching, feature engineering, and DataLoader creation
├── model_architectures.ipynb    # Definitions for Transformer, LSTM, Linear, XGBoost models
├── evaluator.ipynb              # Advanced training (Trainer) and model evaluation (Evaluator) with stress analysis
├── performance_optimizer.ipynb  # Hyperparameter optimization and ensemble weight tuning
├── xai_analyzer.ipynb           # Core XAI logic, SHAP, attention analysis, ethical assessment
├── symbolic_xai.ipynb           # Symbolic Regression implementation and Neural/Symbolic comparison (RQ6)
├── ensemble_model.ipynb         # Confidence-Boosted Ensemble logic
├── requirements.txt             # List of all Python dependencies
├── logs/                        # Output directory for training logs and checkpoints
├── saved_models/                # Output directory for best model files (.pth)
├── results/                     # Output directory for final JSON/CSV reports
├── xai_visualizations/          # Output directory for all XAI plots (.png)
└── symbolic_results/            # Output directory for symbolic regression analysis
```
