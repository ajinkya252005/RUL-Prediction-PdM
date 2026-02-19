# Predictive Maintenance: Remaining Useful Life (RUL) Prediction ⏳🔧

This repository contains a reference implementation for predicting the exact Remaining Useful Life (RUL) of industrial equipment, forming the second predictive pillar of the **Predictive Maintenance: Analysis and Future Outlook** framework.

While Anomaly Detection identifies *if* equipment is behaving abnormally, RUL prediction answers *when* it will fail. This project utilizes a **Long Short-Term Memory (LSTM) Neural Network** to analyze sequential time-series sensor data and predict the continuous countdown to failure.

## 🧠 Approach: The Deep LSTM Method

Predicting equipment failure requires an understanding of how degradation occurs over time. Standard machine learning models look at single moments in time, but our LSTM looks at a continuous "window" of history.

**How it predicts RUL:**
1. **Time-Series Sequences:** We format the sensor data using a sliding window approach (e.g., looking at the last 50 operational cycles). 
2. **Sequential Learning:** The LSTM layers process these 50-cycle blocks to learn the hidden, long-term degradation patterns across 21 different sensors.
3. **Continuous Prediction:** The output layer uses a linear activation function to predict a single, continuous numerical value: the exact number of cycles remaining before the machine breaks down.



## 📊 Dataset: NASA C-MAPSS

This implementation utilizes the industry-standard **NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)**. 

**Structure Used (FD001):**
- `train_FD001.txt`: Run-to-failure data for 100 engines. Used to calculate the RUL target variable and train the LSTM.
- `test_FD001.txt`: Operational data for 100 engines that stops at a random cycle prior to failure.
- `RUL_FD001.txt`: The actual ground-truth remaining cycles for the test engines, used exclusively for final model evaluation.
- **Features:** Engine ID, cycle time, 3 operational settings, and 21 sensor measurements (e.g., temperature, pressure, fan speed).



## 🚀 Getting Started

### Prerequisites
To run the Jupyter/Colab Notebook, ensure you have the following installed:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```
### Usage
1. Clone this repository.
2. Open RUL_Prediction_LSTM.ipynb in Google Colab or Jupyter Notebook.
3. Run the cells sequentially. The notebook is configured to automatically clone the required NASA datasets directly from a GitHub mirror.
4. The code will process the sequences, train the deep learning model, and output the final Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the test data.

### 📈 Results & Evaluation
The final output includes a detailed performance comparison on unseen test engines. It visualizes the Predicted RUL vs. Actual RUL sorted by remaining life, demonstrating the model's ability to accurately predict failure horizons and enable proactive, scheduled maintenance rather than reactive repairs.

_Developed as a reference implementation for advanced Predictive Maintenance frameworks._
