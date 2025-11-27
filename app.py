from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Flask App
app = Flask(__name__)
warnings.filterwarnings("ignore")

# Global Variables
best_model = None
scaler = None
encoders = {}
training_columns = None
original_cols_for_form = None
categorical_cols_map = None
feature_importance_labels = []
feature_importance_scores = []

def train_models_and_prepare_assets():
    """
    Train Random Forest, ANN,
    compare their performance, and select the best one.
    """
    global best_model, scaler, encoders, training_columns, original_cols_for_form, categorical_cols_map
    global feature_importance_labels, feature_importance_scores

    print("\n=== Sleep Quality Prediction Model Training ===")

    # Load data
    try:
        df = pd.read_csv("Sleep_Efficiency.csv")
    except FileNotFoundError:
        print("FATAL ERROR: 'Sleep_Efficiency.csv' not found! The app cannot start.")
        exit()

    # Drop and clean
    df.drop(columns=["ID", "Bedtime", "Wakeup time"], inplace=True, errors='ignore')
    df.fillna(df.median(numeric_only=True), inplace=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Convert efficiency to %
    if df["Sleep efficiency"].max() <= 1:
        df["Sleep efficiency"] *= 100

    # Create target column
    bins = [0, 70, 85, 100]
    labels = ["Poor", "Average", "Good"]
    df["SleepQualityLabel"] = pd.cut(df["Sleep efficiency"], bins=bins, labels=labels, right=True)

    # For form rendering
    original_cols_for_form = df.drop(columns=["Sleep efficiency", "SleepQualityLabel"]).columns.tolist()
    categorical_cols_map = {
        'Gender': df['Gender'].unique().tolist(),
        'Smoking status': df['Smoking status'].unique().tolist()
    }

    # Label encoding
    for c in df.select_dtypes(include=['object', 'category']).columns:
        if c != 'SleepQualityLabel':
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            target_le = LabelEncoder()
            df[c] = target_le.fit_transform(df[c].astype(str))
            encoders[c] = target_le

    X = df.drop(columns=["Sleep efficiency", "SleepQualityLabel"])
    y = df["SleepQualityLabel"]
    training_columns = X.columns

    # Split + balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # -------------------- Random Forest --------------------
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    rf.fit(X_train_scaled, y_train_res)
    y_rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, y_rf_pred)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(7)
    feature_importance_labels = [label.replace('_', ' ').title() for label in top_features.index.tolist()]
    feature_importance_scores = [round(score * 100, 2) for score in top_features.values.tolist()]

    # -------------------- ANN (Neural Network) --------------------
    ann = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y_train_res)), activation='softmax')
    ])
    ann.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-5, verbose=0)

    print("Training Neural Network (ANN)...")
    ann.fit(X_train_scaled, y_train_res, epochs=300, batch_size=8, 
            validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=0)
    y_ann_pred = np.argmax(ann.predict(X_test_scaled), axis=1)
    ann_acc = accuracy_score(y_test, y_ann_pred)
    print(f"ANN Accuracy: {ann_acc:.4f}")

    # -------------------- Comparison --------------------
    print("\n=== Model Comparison ===")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"ANN Accuracy:          {ann_acc:.4f}")

    # Select best model
    best_acc = max(rf_acc, ann_acc)
    if best_acc == ann_acc:
        best_model = ann
        print("\nSelected Model: ANN (best overall performance)")
        print("Interpretation: ANN captured the nonlinear dependencies in the dataset more effectively.")
    else:
        best_model = rf
        print("\nSelected Model: Random Forest (best overall performance)")

    print("\nTraining Complete.")


# --- PREDICTION & SUGGESTION LOGIC ---
def get_personalized_suggestions(user_input_df):
    data_point = user_input_df.iloc[0].copy()
    for col, le in encoders.items():
        if col in data_point and col != 'SleepQualityLabel':
            if data_point[col] in le.classes_:
                data_point[col] = le.transform([data_point[col]])[0]
            else:
                data_point[col] = le.transform([le.classes_[0]])[0]
    for col in training_columns:
        data_point[col] = pd.to_numeric(data_point[col], errors='coerce')
    data_point_processed = pd.DataFrame([data_point], columns=training_columns).fillna(0)
    data_point_scaled = scaler.transform(data_point_processed)

    if isinstance(best_model, Sequential):
        prediction_value = np.argmax(best_model.predict(data_point_scaled), axis=1)[0]
    else:
        prediction_value = int(best_model.predict(data_point_scaled)[0])

    quality_label = encoders['SleepQualityLabel'].inverse_transform([prediction_value])[0]

    suggestions = []
    if quality_label == 'Poor':
        suggestions.append({'text': '😴 Your sleep pattern indicates significant issues. Consider consulting a sleep specialist.'})
        suggestions.append({'text': '📘 Establish a strict, relaxing pre-bedtime routine.'})
    elif quality_label == 'Average':
        suggestions.append({'text': '📈 Your sleep is okay, but can be improved. Focus on consistency.'})
    if float(data_point_processed['Caffeine consumption'].iloc[0]) > 25:
        suggestions.append({'text': '☕ Limit caffeine intake, especially after 2 PM.'})
    if float(data_point_processed['Alcohol consumption'].iloc[0]) > 0:
        suggestions.append({'text': '🍷 Avoid alcohol before bed as it disrupts deep sleep.'})
    if float(data_point_processed['Awakenings'].iloc[0]) > 1:
        suggestions.append({'text': '📱 Reduce evening screen time to minimize awakenings.'})
    if float(data_point_processed['Exercise frequency'].iloc[0]) < 3:
        suggestions.append({'text': '🏃‍♀️ Incorporate regular, moderate exercise into your week.'})
    if not suggestions:
        suggestions.append({'text': '🌟 You have a great sleep routine! Keep it up.'})
        suggestions.append({'text': '🍎 Maintain a balanced diet and stay hydrated for optimal sleep.'})
    return quality_label, prediction_value, suggestions


# --- FLASK ROUTES ---
@app.route("/")
def landing():
    return render_template(
        "landing.html",
        chart_labels=feature_importance_labels,
        chart_scores=feature_importance_scores
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_data = {'quality_label': None, 'prediction_value': None, 'suggestions': None}
    user_inputs = {}
    if request.method == "POST":
        user_inputs = request.form.to_dict()
        user_df = pd.DataFrame([user_inputs])
        quality_label, prediction_value, suggestions = get_personalized_suggestions(user_df)
        prediction_data.update({
            'quality_label': quality_label,
            'prediction_value': prediction_value,
            'suggestions': suggestions
        })
    return render_template(
        "predict.html",
        categorical_cols=categorical_cols_map,
        columns=original_cols_for_form,
        prediction_data=prediction_data,
        user_inputs=user_inputs
    )


# --- RUN APP ---
if __name__ == "__main__":
    train_models_and_prepare_assets()
    app.run(debug=True)
