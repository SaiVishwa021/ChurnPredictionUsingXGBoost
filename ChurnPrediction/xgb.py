from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("Churn_Modelling.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_geography = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Geography'] = le_geography.fit_transform(df['Geography'])

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname', 'HasCrCard'], axis=1)

# Split data into features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Save the trained model and encoders using pickle
with open('xgb_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('le_gender.pkl', 'wb') as file:
    pickle.dump(le_gender, file)
with open('le_geography.pkl', 'wb') as file:
    pickle.dump(le_geography, file)

# Evaluate the model
# Evaluate on training data
y_train_pred = model.predict(X_train)
print("Training Data Classification Report:")
print(classification_report(y_train, y_train_pred))

# Evaluate on test data
y_test_pred = model.predict(X_test)
print("Test Data Classification Report:")
print(classification_report(y_test, y_test_pred))
