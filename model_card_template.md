# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier trained to predict whether an individual earns more than $50K per year based on demographic and employment attributes from the U.S. Census Income dataset. The model uses one‑hot encoding for categorical features and label binarization for the target variable. The training pipeline includes data preprocessing, model training, evaluation, and slice‑based performance analysis.

The model was developed as part of the Udacity Deploying a Scalable ML Pipeline with FastAPI project and is intended for educational purposes.

## Intended Use
The model is intended for:

Demonstrating machine learning deployment using FastAPI

Practicing MLOps concepts such as reproducible pipelines, model serialization, and slice‑based evaluation

Exploring fairness and performance differences across demographic groups

The model is not intended for real‑world decision‑making in hiring, lending, policy, or any domain where misclassification could cause harm.

## Training Data
The training data comes from the Census Income dataset, which contains demographic and employment information such as:

workclass

education

marital-status

occupation

relationship

race

sex

native-country

age

hours-per-week

capital-gain

capital-loss

The target variable is salary, binarized into <=50K and >50K.

An 80/20 train/test split was used.

## Evaluation Data
The evaluation data consists of the held‑out 20% test split from the same Census dataset. The test data was processed using the trained OneHotEncoder and LabelBinarizer to ensure consistent feature representation.

Slice‑based evaluation was also performed across all categorical features to assess fairness and performance variability.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated using precision, recall, and F1 score.

Overall Performance
Precision: 0.7419

Recall: 0.6384

F1 Score: 0.6863

These results indicate that the model performs reasonably well at identifying individuals earning more than $50K, though recall suggests some missed positive cases.

Slice‑Based Performance
Performance varies significantly across demographic slices. Examples include:

High‑performing slices:
Doctorate education: F1 = 0.8793

Masters education: F1 = 0.8409

Exec‑managerial occupation: F1 = 0.7736

Asian‑Pac‑Islander race: F1 = 0.7458

Low‑performing slices:
7th–8th grade education: F1 = 0.0000

Jamaica native-country: F1 = 0.0000

Married‑AF‑spouse marital-status: F1 = 0.0000

Other-service occupation: F1 = 0.3226

Large slices with stable performance:
Private workclass: F1 = 0.6856

White race: F1 = 0.6850

United States native-country: F1 = 0.6814

These results highlight meaningful differences in model performance across demographic groups.

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, and native-country.

Income prediction models can reinforce existing socioeconomic biases if used improperly.

Some demographic slices have very small sample sizes, leading to unstable or misleading metrics.

Misclassification could cause harm if used in real decision systems such as hiring or lending.

This model should never be used for real-world decisions involving individuals.

## Caveats and Recommendations
Retrain the model periodically if used with updated census data.

Consider fairness-aware algorithms for real applications.

Investigate slices with extremely low F1 scores to understand data imbalance.

Evaluate additional metrics such as AUC, confusion matrices, and calibration curves.

Monitor model performance over time if deployed in a production environment.