# 🧭 EDA Analyzer

**Upload. Clean. Visualize. Discover Insights.**

EDA Analyzer is an interactive web-based tool built with **Streamlit** that automates the **Exploratory Data Analysis (EDA)** process.  
It enables users to upload datasets, clean and preprocess data, generate insights through visualizations, and download the cleaned data — all in a sleek, dark-themed dashboard.

---

## 🚀 Features

- 📂 **File Upload:** Supports both `.csv` and `.xlsx` datasets.  
- 🧹 **Automated Data Cleaning:**  
  - Handle missing values automatically  
  - Detect and replace outliers (IQR-based)  
  - Encode categorical columns (Label / One-Hot Encoding)  
  - Scale numerical data (Standardization / Min-Max Scaling)  
- 📊 **Dynamic Visualization:**  
  - Bar Plot, Pie Chart, Box Plot, Histogram, Scatter Plot, Cross Plot  
  - Interactive **Correlation Heatmap (Plotly)**  
  - Frequency table toggle for selected features  
- 🧠 **Automated EDA Report:**  
  - Generated using **ydata_profiling**  
  - Downloadable as an HTML report  
- 💾 **Data Export:**  
  - Download cleaned dataset in **CSV**, **Excel**, or **JSON** format  
- 🌙 **Modern Dark Theme Dashboard:**  
  - Clean UI with sidebar toggle  
  - Smooth transitions and styled controls  

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Frontend & UI** | [Streamlit](https://streamlit.io/) |
| **Data Handling** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly Express](https://plotly.com/python/plotly-express/) |
| **Auto EDA Report** | [YData Profiling (Pandas Profiling)](https://github.com/ydataai/ydata-profiling) |
| **Other** | OpenPyXL, Scikit-learn (for scaling and encoding) |

---


