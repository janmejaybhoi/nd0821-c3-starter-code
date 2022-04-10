import os
import yaml

from ml.data import load_data, process_data
from ml.model import load_model, inference, compute_model_metrics, compute_slice_metrics


def get_categorical_slice_performance(test_data, y_test, y_pred, output_file):
    categorical_features = test_data.select_dtypes(object).columns

    with open(output_file, 'w') as f:
        for cat_feature in categorical_features:
            f.write(f"-------------------------------{cat_feature}-------------------------------\n")
            slice_performance = compute_slice_metrics(
                test_data,
                y_test,
                y_pred,
                cat_feature
            )
            f.write(slice_performance.to_string())
            f.write(f"\n")

    return slice_performance


root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Load data and models
test_data = load_data(root_path, "test_census.csv")
model = load_model(root_path, "model.pkl")
preprocessor = load_model(root_path, "preprocessor.pkl")
label_binarizer = load_model(root_path, "label_binarizer.pkl")

# Read categorical_features
with open(os.path.join(root_path, "starter", "constants.yaml"), 'r') as f:
    categorical_features = yaml.safe_load(f)["categorical_features"]

# Process test data
X_test, y_test, _, _ = process_data(
    test_data, categorical_features=categorical_features, label="salary", training=False,
    preprocessor=preprocessor, label_binarizer=label_binarizer
)

# Apply inference
y_pred = inference(model, X_test)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Print results
print(f"precision={precision}, recall={recall}, fbeta={fbeta}")

# Compute slice performances
slice_performance = get_categorical_slice_performance(
    test_data.loc[:, test_data.columns != 'salary'],
    y_test,
    y_pred,
    'slice_output.txt'
)


# Print slice performance
print(slice_performance)
