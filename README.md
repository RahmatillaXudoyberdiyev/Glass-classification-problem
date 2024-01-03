# Glass Type Classification
This repository contains code for a glass type classification problem using a Random Forest classifier. The goal is to identify the type of glass based on its features. The model is trained on a dataset (\`**glass.csv**\`) and evaluated using various metrics.

## Dataset

The dataset (\`**glass.csv**\`) contains information about different types of glass, with features such as refractive index, sodium, magnesium, aluminum, and more. The target variable is the "Type" of glass.

## Setup

Make sure to install the required dependencies using the \`**requirements.txt**\` file:

```bash
pip install -r requirements.txt
```

## Model Training and Evaluation

1. The dataset is loaded, and features (X) and target variable (y) are extracted.
2. The data is split into training and testing sets (80% training, 20% testing).
3. A Random Forest classifier is trained with default parameters, and its accuracy is recorded.
4. Another Random Forest classifier is trained with tuned parameters, resulting in improved accuracy.
5. Predictions are made on the test set, and various metrics (accuracy, recall, precision, F1 score) are calculated and displayed.

## Model Comparison

The model's performance is compared between the default and tuned configurations:
* **Accuracy (Default):** 0.8372
* **Accuracy (Tuned):** 0.9302

The tuned model shows a significant improvement in accuracy.
## Metrics and Visualizations
The following metrics are displayed and visualized:

* Accuracy
* Recall
* Precision
* F1 Score
* Confusion Matrix

A bar chart is created to visualize the scores, a heatmap for the confusion matrix, and a classification report is also displayed.

## Save Model

The tuned model is saved in two formats using joblib and pickle:

* \`**glass_clf.joblib**\`
* \`**glass_clf.pickle**\`

## Additional Files

* **main.ipynb**: A Jupyter Notebook file containing the main code for model training and evaluation.

## How to Run

1. Ensure Python and the required dependencies are installed.
2. Clone the repository:

```bash
git clone https://github.com/RahmatillaMarvel/Glass-classification-problem.git
cd glass-classification
```
3. Run the main.ipynb notebook or execute the Python script.

```bash
python main.py
```

Feel free to explore and modify the code to experiment with different parameters or datasets.

**Author**: Rahmatilla Xudoyberdiyev

## License

The content of this project is licensed under the [MIT License](LICENSE).

