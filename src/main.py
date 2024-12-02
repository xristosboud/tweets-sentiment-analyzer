import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Custom modules
from preprocessor import RegexPreprocessor
from transformer import EmbeddingTransformer

# Load configuration
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Dataset configuration
dataset_path = config['dataset']['path']
target_column = config['dataset']['target_column']
text_column = config['dataset']['text_column']
class_mapping = config['dataset']['class_mapping']

# Load dataset
df = pd.read_csv(dataset_path)
df = df[[target_column, text_column]]
X = df.drop(columns=target_column, axis=1)
y = df[target_column]

# Split dataset
split_cfg = config['split']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_cfg['test_size'], random_state=split_cfg['random_state']
)

# Embedding models
embedding_models = config['embedding_models']

# Define classifiers
classifiers = []
param_grid = []
for clf_cfg in config['classifiers']:
    clf_type = clf_cfg['type']
    clf_params = clf_cfg['params']
    
    if clf_type == 'XGBoost':
        classifier = xgb.XGBClassifier()
        param_grid.append({
            'embedding__model_name': embedding_models,
            'classifier': [classifier],
            **{f'classifier__{k}': v for k, v in clf_params.items()}
        })
    elif clf_type == 'RandomForest':
        classifier = RandomForestClassifier()
        param_grid.append({
            'embedding__model_name': embedding_models,
            'classifier': [classifier],
            **{f'classifier__{k}': v for k, v in clf_params.items()}
        })

# Define pipeline
pipeline = Pipeline([
    ('regex_preprocessing', RegexPreprocessor(lowercase=True, remove_user_handles=True, remove_hashtags=True, remove_numbers=True)),
    ('embedding', EmbeddingTransformer(model_name='all-MiniLM-L6-v2')),
    ('classifier', None)
])

# Grid search configuration
grid_cfg = config['grid_search']
grid_search = GridSearchCV(pipeline, param_grid, **grid_cfg)
grid_search.fit(X_train[text_column], y_train)

# Evaluate
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test[text_column])

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(classification_report(y_test, y_pred, target_names=list(class_mapping.keys())))