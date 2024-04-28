import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the dataset
data = pd.read_csv('/Users/hamdilibanahmed/Downloads/ML Model/diabetes_dataset.csv')

# Handling Zeros
for column in ['Age', 'BMI', 'Blood Glucose (mg/dL)']:
    data[column] = data[column].replace(0, data[column].median())

# Split the dataset into features and target variable
X = data.drop('Diabetes Diagnosis', axis=1)
y = data['Diabetes Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Define the preprocessing steps for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Model Selection and Training
# Logistic Regression
logreg_model = LogisticRegression(random_state=42, solver='liblinear')

# Define the hyperparameters to tune for Logistic Regression
logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']  
}

# Perform grid search for hyperparameter tuning
logreg_grid_search = GridSearchCV(logreg_model, logreg_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
logreg_grid_search.fit(X_train_processed, y_train)

# Print the best parameters found by grid search
print("Best Parameters: ", logreg_grid_search.best_params_)

# Train the model with the best parameters
logreg_best = LogisticRegression(random_state=42, solver='liblinear', C=logreg_grid_search.best_params_['C'], penalty=logreg_grid_search.best_params_['penalty'])
logreg_best.fit(X_train_processed, y_train)

# Make predictions on the test set
y_pred = logreg_best.predict(X_test_processed)

# Evaluate the model
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-score: ", f1_score(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))

# Cross-validation
cross_val_scores = cross_val_score(logreg_best, X_train_processed, y_train, cv=5, scoring='roc_auc')
print("Cross-validation ROC AUC scores: ", cross_val_scores)
print("Mean cross-validation ROC AUC score: ", cross_val_scores.mean())

# Save the model
joblib.dump(logreg_best, 'diabetes_model.pkl')
# Save the encoders and scaler
