# src/kedro_project/pipelines/data_processing/scripts/data_card_generator.py

import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from datetime import datetime
from openml import datasets

def data_card_generator(dataset_id: int, output_path: str) -> None:
    # Get dataset
    dataset = datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    df = pd.DataFrame(X, columns=attribute_names)
    df[dataset.default_target_attribute] = y

    # Save dataset
    df.to_csv('data/soybean_v1.csv', index=False)

    # Obtain dataset information
    results = {
        'nr_instances': df.shape[0],
        'nr_attributes': df.shape[1] - 1,  # Excluding the target column
        'attr_to_inst': (df.shape[1] - 1) / df.shape[0],
        'nr_classes': df[dataset.default_target_attribute].nunique(),
        'freq_class': df[dataset.default_target_attribute].value_counts().max() / len(df),
        'avg_categories': df.drop(dataset.default_target_attribute, axis=1).nunique().mean(),
        'max_categories': df.drop(dataset.default_target_attribute, axis=1).nunique().max(),
        'min_categories': df.drop(dataset.default_target_attribute, axis=1).nunique().min(),
        'most_common_value': df.drop(dataset.default_target_attribute, axis=1).mode().iloc[0].mode()[0]
    }

    # Extract MFE features
    safe_mfe_features = ['class_ent', 'mut_inf']
    for feature in safe_mfe_features:
        try:
            mfe = MFE(features=[feature], random_state=42)
            X_values = df.drop(dataset.default_target_attribute, axis=1).values
            y_values = df[dataset.default_target_attribute].values
            mfe.fit(X_values, y_values)
            ft = mfe.extract()
            results[ft[0][0]] = ft[1][0]
        except Exception as e:
            print(f"Couldn't extract {feature}: {str(e)}")

    # Prepare data card content
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    features_info = dataset.features
    features_data = {
        'Variable Name': [],
        'Role': [],
        'Type': [],
    }
    target_name = dataset.default_target_attribute

    for index, feature in features_info.items():
        feature_name = feature.name
        feature_type = 'Categorical' if feature.data_type == 'nominal' else 'Numerical'
        role = 'Target' if feature_name == target_name else 'Feature'
        features_data['Variable Name'].append(feature_name)
        features_data['Role'].append(role)
        features_data['Type'].append(feature_type)

    features_df = pd.DataFrame(features_data)
    table_str = features_df.to_markdown(index=False)

    data_card_content = f"""# Data Card: Soybean Disease Dataset
    - **Dataset ID**: {dataset.id}
    - **Version**: {dataset.version}
    - **This document was last updated on**: {current_date}
    - **Link to OpenML page**: {dataset.openml_url}
    ## Description
    {dataset.description}
    ## Variables Table
    {table_str}
    ## Dataset Information
    - **Number of Instances**: {results['nr_instances']}
    - **Number of Attributes**: {results['nr_attributes']}
    - **Attribute to Instance Ratio**: {results['attr_to_inst']:.2f}
    - **Number of Classes**: {results['nr_classes']}
    - **Frequency of the Most Common Class**: {results['freq_class']:.2%}
    - **Average Number of Categories per Attribute**: {results['avg_categories']:.2f}
    - **Maximum Number of Categories in an Attribute**: {results['max_categories']}
    - **Minimum Number of Categories in an Attribute**: {results['min_categories']}
    - **Most Common Value in Attributes**: {results['most_common_value']}
    ### Extracted Meta-Features
    - **Class Entropy**: {results.get('class_ent', 'Not Available')}
    - **Mutual Information**: {results.get('mut_inf', 'Not Available')}
    """

    # Save the data card to a file
    with open(output_path, 'w') as file:
        file.write(data_card_content)
