import json

def generate_data_card():
    with open('data/08_reporting/data-card/data_card.json', 'r') as f:
        data_card = json.load(f)
        data_card = json.loads(data_card)

    content = f"""
# Data Card: {data_card['dataset_name']}

## Dataset Information
- **Name**: {data_card['dataset_name']}
- **Description**: Credit risk assessment dataset
- **Version**: 1.0

## Dataset Characteristics
- **Number of Instances**: {data_card['number_of_rows']}
- **Number of Features**: {data_card['number_of_features']}
- **Target Variable**: y (boolean)

## Data Quality Metrics
- **Number of Features**: {data_card['data_quality_metrics']['num_features']}
- **Number of Rows**: {data_card['data_quality_metrics']['num_rows']}
- **Missing Values**: {data_card['data_quality_metrics']['missing_values']}%

## Features
{', '.join(data_card['feature_names'])}

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
- The dataset is relatively small ({data_card['number_of_rows']} instances), which may limit its representativeness.
- Some categorical variables may have imbalanced categories.
- The additional numerical features (X_1 to X_10) lack clear descriptions of what they represent.
    """
    
    with open('docs-quarto/data_card_content.md', 'w') as f:
        f.write(content)

def generate_model_card():
    with open('data/08_reporting/model-card/model_card.json', 'r') as f:
        model_card = json.load(f)
        model_card = json.loads(model_card)

    content = f"""

## Model Details
- **Model Type**: {model_card['model_type']}

## Intended Use
- **Primary Use**: Credit risk assessment
- **Intended Users**: Financial institutions, credit analysts

## Model Architecture
- **Base Estimators**: {model_card['model_parameters']['n_estimators']/10} decision trees
- **Max Depth**: {model_card['model_parameters']['max_depth']}
- **Criterion**: {model_card['model_parameters']['criterion']}

## Model Parameters
{json.dumps(model_card['model_parameters'], indent=2)}

## Performance Metrics
- **Accuracy**: {model_card['evaluation_metrics']['accuracy']:.2f}
- **Precision**: {model_card['evaluation_metrics']['precision']:.2f}
- **Recall**: {model_card['evaluation_metrics']['recall']:.2f}
- **F1 Score**: {model_card['evaluation_metrics']['f1']:.2f}

## Training Data
- **Dataset**: dataset_id_96
- **Splitting Method**: Random split (80% training, 20% testing)
- **Preprocessing**: Standard scaling for numerical features, one-hot encoding for categorical features

## Ethical Considerations
- Decisions based on this model's output should be explainable and challengeable.
- The model should be used in compliance with relevant financial regulations and data protection laws.

## Caveats and Recommendations
- The model's performance may vary for different subgroups. It's recommended to evaluate the model's fairness across various demographic groups.
- Regular retraining is advised to ensure the model remains accurate as financial trends evolve.
- The model should be used in conjunction with other risk assessment methods and human judgment.
    """
    with open('docs-quarto/model_card_content.md', 'w') as f:
        f.write(content)


if __name__ == "__main__":
    generate_data_card()
    generate_model_card()
    print("Data card and Model card content generated successfully.")