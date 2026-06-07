# Microchannel Flow Neural Operator

This Streamlit application analyzes Computational Fluid Dynamics (CFD) data and visualizes predictions from trained neural network models. The app allows users to upload `.dat` files, run predictions, and compare the results against ground truth data.

## Features

- **Upload and Parse CFD Data**: Load `.dat` files for analysis.
- **Model Predictions**: Utilize trained U-Net AM, T-Net, and U-Net models to predict fluid flow parameters.
- **Visualization**: Generate contour plots for predicted and actual values, including error metrics.
- **Performance Metrics**: Display evaluation metrics such as RMSE, MAE, and R² scores for each model.
- **Fluid Flow Summary**: Provide a narrative summary of the flow patterns based on predictions.

## Requirements

- Python 3.7 or higher
- Required Python packages are listed in `requirements.txt`.

## Installation

1. Clone this repository:

   ```bash
   git clone <your_github_repo_url>
   cd your_project
