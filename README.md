# Emoji Prediction From Text
Group 12 - Imran Ahmed, Defne Aydemir, Tina Shi

## Project Overview

Emojis play an important role in digital communication by conveying emotion, tone, and context beyond plain text. This project treats emojis as meaningful linguistic signals and aims to predict the most likely emoji following a given tweet. We train various models, namely KNN, logistic regression, SVM, zero-shot LLM, and transformer-based architectures like RoBERTa, DeBERTa, and BERTweet. All experiments are implemented in Python using common data science and machine learning libraries.

Our dataset is from [Kaggle](https://www.kaggle.com/datasets/ericwang1011/tweets-with-emoji?resource=download).

## **Instructions**
1. Clone this repo
```
git clone git@github.com:tinas1494/ds301_group12.git
```
2. Move into the directory of the repo
```
cd ds301_group12/
```
3. Run `proposal_eda.ipynb`
```
jupyter notebook proposal_eda.ipynb
```
or
```
jupyter lab proposal_eda.ipynb
```
- This will create the `combined_data.csv` file (too large to be uploaded to GitHub) used in our models.
4. All models and dependencies are implemented in the Jupyter notebooks and can be run directly.


### Troubleshooting
- If `combined_data.csv` is missing, rerun `proposal_eda.ipynb`
- If notebooks fail to run, restart the kernel and rerun all cells

## Repo Structure
The `data/` folder contains 43 CSV files corresponding to 43 emojis, with 20,000 tweet samples each. Running `proposal_eda.ipynb` combines these CSV files into `combined_data.csv`, with some additional columns useful for model training and analysis.

The `notebooks/` folder contains all our notebooks, which is further organized into `notebooks/baseline/` and `notebooks/deep_learning/`.
- `notebooks/baseline/` contains our baseline, simple machine learning models: KNN, logistic regression, and SVM.
- `notebooks/deep_learning/` contains our improved models: BERTweet, LLM, RoBERTa, and DeBERTa.
    - `improved_bertweet.ipynb` is BERTweet with grouped emojis labels method.

## Results

**Accuracy for our models**
KNN: 0.15
Logistic Regression: 0.20
SVM: 0.25
Zero-Shot LLM: 0.20
DeBERTa: 0.13
RoBERTa: 0.16
BERTweet: 0.28
Improved BERTweet: 0.30

Emoji prediction is a challenging task due to the nuanced and context-dependent nature of emoji usage. Many emojis are interchangeable or convey similar emotions, and they are often used ironically or in rapidly evolving social contexts. Tweets are typically short and provide limited textual context, which makes disambiguating the intended emoji even harder.

Baseline models (KNN, logistic regression, SVM) achieve modest performance. Transformer-based models generally perform better, with BERTweet and improved BERTweet models achieving the highest accuracies. Surprisingly, DeBERTa and RoBERTa perform worse than BERTweet on this dataset, which could be due to the short and noisy nature of tweets, or insufficient fine-tuning. Overall, emoji prediction is challenging, and small gains in accuracy reflect the subtle and context-dependent usage of emojis in social media text.