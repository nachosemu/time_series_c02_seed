# Time Series Transformer - CO2 Seed Project

This project is focused on time series forecasting using a Time Series Transformer. The pipeline is designed to handle CO2 intensity data and supports training, validation, and prediction.

## Installation

To get started, first clone the repository and ensure you have Python installed (version 3.8 or higher is recommended).

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/time_series_c02_seed.git
    cd time_series_c02_seed
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Follow the steps provided in the Jupyter Notebook tutorial:

1. Open the notebook:
    ```bash
    jupyter notebook transformer_eu.ipynb
    ```
   
2. Run the cells sequentially to understand the pipeline and experiment with the data and models.

## Project Structure

- **`transformer_eu.ipynb`**: Main tutorial notebook to understand and experiment with the pipeline.
- **`scr/`**: Contains custom modules:
  - `c02_intensity_data_loader.py`: Handles data loading and preprocessing.
  - `ts_trans_model.py`: Defines the Transformer model for time series forecasting.
- **`requirements.txt`**: Lists all required Python libraries for the project.
- **`README.md`**: This file, describing the project setup and usage.

## Features

- Utilizes **Hugging Face Transformers** for time series forecasting.
- Supports CO2 intensity data preprocessing with **GluonTS** transformations.
- Includes example data loading, training, test, and inference workflows.

---

Enjoy experimenting with the time series transformer model and customizing it for your use case!
