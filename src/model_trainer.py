import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.data_handler import load_data, get_train_test_data

def train_and_export_model(model_dir='models'):
    """
    Trains the optimal Logistic Regression model and exports it along with the scaler.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("🔄 Loading data...")
    X, y = load_data()
    
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("🤖 Training model...")
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("💾 Exporting artifacts...")
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Optional: Export the feature names so the app knows the exact expected order
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_names.pkl'))
    
    # Calculate feature importances using Logistic Regression coefficients (absolute values)
    importances = abs(model.coef_[0])
    feature_importance_dict = dict(zip(X.columns, importances))
    joblib.dump(feature_importance_dict, os.path.join(model_dir, 'feature_importances.pkl'))

    print("✅ Model, scaler, and metadata exported successfully!")

if __name__ == "__main__":
    train_and_export_model()
