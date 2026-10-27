# Google Play Store App Popularity Prediction

## Project Overview
This project analyses Google Play Store applications and predicts app popularity by estimating the logarithm of installs using regression models. It includes data collection, preprocessing, exploratory data analysis, statistical analysis, model evaluation, and an interactive Gradio interface.

## Objectives
- Collect application data using `google-play-scraper`.
- Clean and prepare the dataset for analysis.
- Explore relationships between application features and installs.
- Train and evaluate regression models.
- Provide an interactive interface for estimating app installs.

## Technologies Used
- Python 3.13.16
- Pandas and NumPy
- SciPy and Statsmodels
- Scikit-learn
- Matplotlib
- google-play-scraper
- Gradio
- Google Colab

## Models
### Simple Linear Regression
Uses the logarithm of review count (`log_reviews`) to predict the logarithm of installs (`log_installs`).

### Multiple Linear Regression
Uses multiple application attributes, including review count, rating, free status, category, and content rating, to predict `log_installs`.

## Model Evaluation

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| Simple Linear Regression | 0.803 | 1.217 | 0.907 |
| Multiple Linear Regression | 0.830 | 1.130 | 0.833 |

Multiple Linear Regression performed better on the reported test-set metrics. The metrics are measured on the log-transformed install target.

> Confirm that these values match the final version of your notebook before publishing.

## Repository Structure

```text
google-play-store-app-popularity-prediction/
├── SI_LAB_PROJECT_1.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── screenshots/
```

The notebook filename above is an example; update it if your notebook has a different name.

## How to Run
1. Clone or download this repository.
2. Install the packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in Google Colab or Jupyter Notebook.
4. Run the notebook cells in order.

Some data collection cells may require an internet connection.

## Limitations
- Data collected from search results may be affected by sampling bias.
- Install counts may be reported as estimates or ranges rather than exact counts.
- Predictions are limited by the features available in the dataset.

## Future Improvements
- Collect a larger and more representative dataset.
- Add further app and developer features.
- Improve feature engineering and model evaluation.
- Deploy the prediction interface as a web application.

## Author
Kushal Trivedi
