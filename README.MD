# Car Price Guesser

## Overview

Car Price Guesser is a streamlined Python tool that predicts used car resale values in **Indian Rupees (₹)** using a trained regression model. To run it simply click twice on the RunApp.bat

## Features

* Data loading and preprocessing for common car features (make, model, year, kilometers driven, etc.)
* Model training script to build or update prediction model
* Command-line interface for quick price estimates
* A Simple Bat file to Run The App and download requirments 

## Prerequisites

* Python 3.8 or higher
* Git (to clone this repository)

## Installation

> **Note:** Make sure your virtual environment is activated. All commands below should be run from within the venv.

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/car-price-guesser.git
   cd car-price-guesser
   ```
2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   . .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Train the Model

```bash
python train_model.py \
  --data data/car_data.csv \
  --output models/car_price_model.pkl
```

### Predict Prices

```bash
python predict.py \
  --model models/car_price_model.pkl \
  --make Maruti --model Alto --year 2015 --km_driven 45000
```

## Project Structure

```
car-price-guesser/
├── data/                        # CSV datasets (e.g., car_data.csv)
├── models/                      # Trained model artifacts
│   └── car_price_model.pkl
├── train_model.py              # Script to train regression model
├── predict.py                  # Script to generate price predictions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, reach out at **[malshkeili01@gmail.com](mailto:malshkeili01@gmail.com)**.
