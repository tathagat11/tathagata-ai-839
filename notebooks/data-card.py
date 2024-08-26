#!/usr/bin/env python
# coding: utf-8

# #### Import libraries

# In[15]:


import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from datetime import datetime
from openml import datasets


# #### Getting dataset

# In[16]:


dataset = datasets.get_dataset(42) # This also contains metadata regarding the dataset like version, description, etc.
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

df = pd.DataFrame(X, columns=attribute_names)
df[dataset.default_target_attribute] = y

# Saving the dataset (in data directory)
df.to_csv('data/soybean_v1.csv', index=False)


# #### Obtain some dataset information

# In[17]:


results = {}

results['nr_instances'] = df.shape[0]
results['nr_attributes'] = df.shape[1] - 1  # Excluding the target column
results['attr_to_inst'] = (df.shape[1] - 1) / df.shape[0]

# Class information
results['nr_classes'] = df[dataset.default_target_attribute].nunique()
results['freq_class'] = df[dataset.default_target_attribute].value_counts().max() / len(df)

# Categorical data specific information
results['avg_categories'] = df.drop(dataset.default_target_attribute, axis=1).nunique().mean()
results['max_categories'] = df.drop(dataset.default_target_attribute, axis=1).nunique().max()
results['min_categories'] = df.drop(dataset.default_target_attribute, axis=1).nunique().min()

# Most common values
results['most_common_value'] = df.drop(dataset.default_target_attribute, axis=1).mode().iloc[0].mode()[0]

# Try to extract some MFE features individually
safe_mfe_features = ['class_ent', 'mut_inf']
for feature in safe_mfe_features:
    try:
        mfe = MFE(features=[feature], random_state=42)
        X_values = df.drop(dataset.default_target_attribute, axis=1).values
        y_values = df[dataset.default_target_attribute].values  # Convert to numpy array
        mfe.fit(X_values, y_values)
        ft = mfe.extract()
        results[ft[0][0]] = ft[1][0]
    except Exception as e:
        print(f"Couldn't extract {feature}: {str(e)}")

meta_features = pd.DataFrame(list(results.items()), columns=['Feature', 'Value'])


# In[18]:


current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Extract feature information
features_info = dataset.features

# Prepare a DataFrame to display the information
features_data = {
    'Variable Name': [],
    'Role': [],
    'Type': [],
}

target_name = dataset.default_target_attribute

for index, feature in features_info.items():
    feature_name = feature.name  # Extract the feature name
    
    # Determine if the feature is categorical or numerical
    feature_type = 'Categorical' if feature.data_type == 'nominal' else 'Numerical'
    
    # Determine the role (Target or Feature)
    role = 'Target' if feature_name == target_name else 'Feature'

    # Append data to the lists
    features_data['Variable Name'].append(feature_name)
    features_data['Role'].append(role)
    features_data['Type'].append(feature_type)


features_df = pd.DataFrame(features_data)

# Convert the DataFrame to a Markdown-formatted table
table_str = features_df.to_markdown(index=False)

# Or manually if you don't have Pandas >=1.3 (which includes `to_markdown`)
table_str = f"{' | '.join(features_df.columns)}\n" + \
            f"{' | '.join(['---'] * len(features_df.columns))}\n" + \
            "\n".join([' | '.join(map(str, row)) for row in features_df.values])


data_card_content = f"""# Data Card: Soybean Disease Dataset
- **Dataset ID**: {dataset.id}
- **Version**: {dataset.version}
- **This document was last updated on**: {current_date}{dataset.openml_url}
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
"""

# - **Mutual Information**: {results.get('mut_inf', 'Not Available')}

# Display or save the data card content
print(data_card_content)

# Optionally save to a file
with open('docs/data-card.md', 'w') as file:
    file.write(data_card_content)

