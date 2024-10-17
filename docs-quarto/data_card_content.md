
# Data Card: dataset_id_96

## Dataset Information
- **Name**: dataset_id_96
- **Description**: Credit risk assessment dataset
- **Version**: 1.0

## Dataset Characteristics
- **Number of Instances**: 100
- **Number of Features**: 32
- **Target Variable**: y (boolean)

## Data Quality Metrics
- **Number of Features**: 32
- **Number of Rows**: 100
- **Missing Values**: 0.0%

## Features
checking_status, duration, credit_history, purpose, credit_amount, savings_status, employment, installment_commitment, personal_status, other_parties, residence_since, property_magnitude, age, other_payment_plans, housing, existing_credits, job, num_dependents, own_telephone, foreign_worker, health_status, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, y

## Feature Descriptions
- **checking_status**: Status of existing checking account
- **credit_history**: Credit history of the applicant
- **purpose**: Purpose of the credit
- **savings_status**: Status of savings accounts/bonds
- **employment**: Present employment since
- **personal_status**: Sex & marital status
- **other_parties**: Other debtors / guarantors
- **property_magnitude**: Property (e.g. real estate)
- **other_payment_plans**: Other installment plans
- **housing**: Housing (own, rent, or free)
- **job**: Job type
- **foreign_worker**: Is the applicant a foreign worker?
- **health_status**: Health status of the applicant
- **y**: Boolean indicating credit risk (True = good, False = bad)

## Data Collection
- **Method**: Unknown

## Intended Use
This dataset is intended for credit risk assessment. It can be used to train machine learning models to predict the likelihood of credit default.

## Ethical Considerations
- Ensure fair and unbiased use of the data, particularly regarding protected attributes like personal status.
- Be cautious of potential biases in the original data collection process.
- Consider the implications of using this data for decision-making in financial contexts.

## Known Limitations
- The dataset is relatively small (100 instances), which may limit its representativeness.
- Some categorical variables may have imbalanced categories.
- The additional numerical features (X_1 to X_10) lack clear descriptions of what they represent.
    