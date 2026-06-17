import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from data_loader import load_data, get_match_data
from preprocessor import preprocess, encode_features, get_features

df = load_data('IPL.csv')
df = get_match_data(df)
df = preprocess(df)
df, encoders = encode_features(df)
X, y = get_features(df)

# 80% for training 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# trying 3 different models to see which works best
models = {
    'logreg': LogisticRegression(max_iter=1000),
    'rf': RandomForestClassifier(n_estimators=150, max_depth=8),
    'xgb': XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='mlogloss')
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, m.predict(X_test))
    print(f'{name} -> {results[name]:.4f}')

# picking the model with highest accuracy
best = max(results, key=results.get)
print(f'\nbest model: {best}')

# saving model and encoders for later use
joblib.dump(models[best], 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')

joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')