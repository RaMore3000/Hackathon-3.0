
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load train & test
train = pd.read_csv('Train_Data.csv').dropna(subset=['age_group'])
train['age_group'] = train['age_group'].map({'Adult': 0, 'Senior': 1})
X_train = train.drop(columns=['SEQN', 'age_group'])
y_train = train['age_group']
test = pd.read_csv('Test_Data.csv')
X_test = test.drop(columns=['SEQN'])

# Choose classifier: RF or balanced random forest
clf = RandomForestClassifier(
    n_estimators=10,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# Or use BalancedRandomForestClassifier
# clf = BalancedRandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', clf)
])

# Cross-validation on train set
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("CV accuracy:", scores, "Mean:", scores.mean())

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print("Test predictions distribution:", pd.Series(preds).value_counts())

# Save submission as a CSV file
pd.DataFrame({'age_group': preds.astype(int)}).to_csv(
    'submission.csv', index=False)
