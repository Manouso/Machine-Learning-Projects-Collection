# Advertising Data EDA 

This project performs an Explanatory Data Analysis (EDA) on an Advertising dataset from Kaggle
to understand and compare how different marketing channels influence Sales

---

## Project Structure

- **data-eda/**
  - `eda_project.ipynb` — Main notebook for experiments  
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Objectives

- Load and clean the dataset
- Perform descriptive statistics and correlation analysis
- Visualise the relationship between advertising budgets and sales
- Identify which marketing channels drive the highest impact on sales

---

## Dataset

- Source:  [Advertising Dataset on Kaggle](https://www.kaggle.com/datasets/ashydv/advertising-dataset)
- Columns:
    - `TV`: Advertising spend on TV
    - `Radio`: Advertising spend on Radio
    - `Newspaper`: Advertising spend on Newspapers
    - `Sales`: Sales revenue 
---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Insights

- TV advertising exhibits the strongest positive correlation with Sales (≈0.9), indicating that increased TV spending is consistently associated with higher revenue.

- Radio advertising shows a moderate positive relationship with Sales (≈0.35), suggesting it contributes decently but less powerfully than TV.

- Newspaper advertising demonstrates a weak correlation with Sales (≈0.16), implying minimal impact on revenue performance.

- Newspaper spend is right-skewed, indicating that only a few instances involve very high investment in this channel.

- The distribution of Sales is approximately normal, showing no significant outliers.

- Overall, the analysis suggests that TV and Radio are the most effective advertising channels for driving sales, while Newspaper contributes the least.
