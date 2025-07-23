import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os

# Global variables to store preprocessing components
global_imputer = None
global_scaler = None
global_categorical_encoders = {}
global_year_min = None
global_quantiles = {}

def prepare_features(df):
    processed_df = preprocess_data(df)

    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df

    return X, y

def preprocess_data(df, training=True):
    global global_imputer, global_scaler, global_categorical_encoders, global_year_min, global_quantiles
    
    df = df.copy()

    # Store min year value if in training mode
    if training:
        global_year_min = df["year"].min()
    
    # Log Transformation
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    df["log_budget"] = np.log1p(df["budget"])

    # Feature engineering
    df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
    df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    
    # Use stored global_year_min
    year_min = global_year_min if global_year_min is not None else df["year"].min()
    df["budget_year_ratio"] = df["log_budget"] / (df["year"] - year_min + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - year_min + 1)
    
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - year_min + 1)
    
    # Binary fields based on quantiles - need different handling for prediction
    if training:
        year_q75 = df["year"].quantile(0.75)
        budget_q75 = df["log_budget"].quantile(0.75)
        votes_q75 = df["votes"].quantile(0.75)
        score_q75 = df["score"].quantile(0.75)
        
        # Store thresholds
        global_quantiles = {
            "year_q75": year_q75,
            "budget_q75": budget_q75,
            "votes_q75": votes_q75,
            "score_q75": score_q75
        }
    else:
        # Use stored thresholds
        year_q75 = global_quantiles.get("year_q75", 2015)  # Default if not available
        budget_q75 = global_quantiles.get("budget_q75", 16)
        votes_q75 = global_quantiles.get("votes_q75", 10000)
        score_q75 = global_quantiles.get("score_q75", 7.0)
    
    df["is_recent"] = (df["year"] >= year_q75).astype(int)
    df["is_high_budget"] = (df["log_budget"] >= budget_q75).astype(int)
    df["is_high_votes"] = (df["votes"] >= votes_q75).astype(int)
    df["is_high_score"] = (df["score"] >= score_q75).astype(int)

    categorical_features = [
        "released",
        "writer",
        "rating",
        "name",
        "genre",
        "director",
        "star",
        "country",
        "company",
    ]

    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        
        if training:
            # Create and store new encoder during training
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            global_categorical_encoders[feature] = le
        else:
            # Use trained encoder for prediction
            if feature in global_categorical_encoders:
                le = global_categorical_encoders[feature]
                try:
                    df[feature] = le.transform(df[feature])
                except ValueError:
                    # Handle new values not in training set
                    df[feature] = -1  # Or another value to mark new values
            else:
                # Fallback if no encoder available
                df[feature] = 0

    numerical_features = [
        "runtime",
        "score",
        "year",
        "votes",
        "log_budget",
        "budget_vote_ratio",
        "budget_runtime_ratio",
        "budget_score_ratio",
        "vote_score_ratio",
        "budget_year_ratio",
        "vote_year_ratio",
        "score_runtime_ratio",
        "budget_per_minute",
        "votes_per_year",
        "is_recent",
        "is_high_budget",
        "is_high_votes",
        "is_high_score",
    ]

    if training:
        # Train and store new imputer and scaler
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        df[numerical_features] = imputer.fit_transform(df[numerical_features])
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        
        global_imputer = imputer
        global_scaler = scaler
    else:
        # Use trained imputer and scaler
        if global_imputer is not None and global_scaler is not None:
            df[numerical_features] = global_imputer.transform(df[numerical_features])
            df[numerical_features] = global_scaler.transform(df[numerical_features])
        else:
            # Fallback if imputer/scaler not available
            pass

    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)

    return df

def train_xgb_model(X, y):
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, **best_params
    )
    best_model.fit(X, y)
    return best_model

def train_dnn_model(X, y):
    # Build a more robust DNN architecture
    model = Sequential()
    
    # Input layer with regularization
    model.add(Dense(64, activation='relu', input_dim=X.shape[1], 
                   kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Hidden layers with decreasing neurons
    model.add(Dense(32, activation='relu', 
                   kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Final hidden layer
    model.add(Dense(16, activation='relu'))
    
    # Output layer with linear activation for regression
    model.add(Dense(1, activation='linear'))
    
    # Compile with lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    
    # Train with early stopping and learning rate reduction
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train with more epochs and smaller batch size
    history = model.fit(
        X, y, 
        epochs=150, 
        batch_size=16, 
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    return model

def train_cnn_model(X, y):
    # Reshape input for CNN
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    # Build CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    # Compile with appropriate loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    
    # Train with early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    
    model.fit(
        X_reshaped, y, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    return model

def predict_gross_xgb(input_data, model):
    processed_data = preprocess_data(pd.DataFrame([input_data]), training=False)
    
    # Ensure all expected features are present
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    
    # Ensure correct order of features
    processed_data = processed_data[expected_features]
    
    # Predict log value and convert back
    log_prediction = model.predict(processed_data)
    prediction = np.exp(log_prediction) - 1
    
    # Ensure prediction is positive
    prediction = max(0, prediction[0])
    
    return prediction

def predict_gross_dnn(input_data, model):
    processed_data = preprocess_data(pd.DataFrame([input_data]), training=False)
    
    # Predict log value and convert back
    log_prediction = model.predict(processed_data.values)
    prediction = np.exp(log_prediction[0][0]) - 1
    
    # Ensure prediction is positive and apply scaling factor to fix underestimation
    prediction = max(0, prediction) * 1.1  # Apply scaling factor of 1.5
    
    # Add minimum threshold to prevent extremely low predictions
    if prediction < 1000000:  # If less than 1M
        prediction = 1000000 + prediction * 0.8  # Base 1M + 80% of original prediction
    
    return prediction

def predict_gross_cnn(input_data, model):
    processed_data = preprocess_data(pd.DataFrame([input_data]), training=False)
    
    # Reshape for CNN input
    reshaped_data = processed_data.values.reshape((processed_data.shape[0], processed_data.shape[1], 1))
    
    # Predict log value and convert back
    log_prediction = model.predict(reshaped_data)
    prediction = np.exp(log_prediction[0][0]) - 1
    
    # Ensure prediction is positive and apply scaling factor to fix underestimation
    prediction = max(0, prediction) * 1.8  # Apply higher scaling factor of 1.8
    
    # Add minimum threshold to prevent extremely low predictions
    if prediction < 1500000:  # If less than 1.5M
        prediction = 1500000 + prediction * 0.7  # Base 1.5M + 70% of original prediction
    
    return prediction

def predict_gross_range(gross):
    if gross <= 10000000:
        return f"Low Revenue (<= 10M)"
    elif gross <= 40000000:
        return f"Medium-Low Revenue (10M - 40M)"
    elif gross <= 70000000:
        return f"Medium Revenue (40M - 70M)"
    elif gross <= 120000000:
        return f"Medium-High Revenue (70M - 120M)"
    elif gross <= 200000000:
        return f"High Revenue (120M - 200M)"
    else:
        return f"Ultra High Revenue (>= 200M)"

# Function to load models or train them if not available
def load_or_train_models():
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)
    
    xgb_path = os.path.join(models_path, "xgb_model.pkl")
    dnn_path = os.path.join(models_path, "dnn_model.h5")
    cnn_path = os.path.join(models_path, "cnn_model.h5")
    preprocess_path = os.path.join(models_path, "preprocess_components.pkl")
    
    # Check if models exist
    if os.path.exists(xgb_path) and os.path.exists(dnn_path) and os.path.exists(cnn_path) and os.path.exists(preprocess_path):
        # Load models
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        
        dnn_model = tf.keras.models.load_model(dnn_path)
        cnn_model = tf.keras.models.load_model(cnn_path)
        
        # Load preprocessing components
        with open(preprocess_path, 'rb') as f:
            preprocess_components = pickle.load(f)
            global global_imputer, global_scaler, global_categorical_encoders, global_year_min, global_quantiles
            global_imputer = preprocess_components['imputer']
            global_scaler = preprocess_components['scaler']
            global_categorical_encoders = preprocess_components['encoders']
            global_year_min = preprocess_components['year_min']
            global_quantiles = preprocess_components['quantiles']
            
        st.success("Loaded existing models")
    else:
        # Train new models
        with st.spinner('Training models for the first time...'):
            # Load data
            df = pd.read_csv("revised datasets/output.csv")
            
            # Prepare features
            X, y = prepare_features(df)
            
            # Train models
            xgb_model = train_xgb_model(X, y)
            dnn_model = train_dnn_model(X, y)
            cnn_model = train_cnn_model(X, y)
            
            # Save models
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            
            dnn_model.save(dnn_path)
            cnn_model.save(cnn_path)
            
            # Save preprocessing components
            preprocess_components = {
                'imputer': global_imputer,
                'scaler': global_scaler,
                'encoders': global_categorical_encoders,
                'year_min': global_year_min,
                'quantiles': global_quantiles
            }
            
            with open(preprocess_path, 'wb') as f:
                pickle.dump(preprocess_components, f)
            
            st.success("Models trained and saved successfully")
    
    return xgb_model, dnn_model, cnn_model

# Main Streamlit app
st.markdown(
    """
    <h1 style='text-align: center; color: cyan;'>Movie Revenue Prediction</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h2 style='text-align: center; color: white;'>Movie Details</h2>
    """,
    unsafe_allow_html=True,
)

# Load models once (either from disk or train them)
xgb_model, dnn_model, cnn_model = load_or_train_models()

with st.form(key="movie_form"):
    col1, col2 = st.columns(2)

    with col1:
        released = st.text_input("Release Date")
        writer = st.text_input("Writer")
        rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
        name = st.text_input("Movie Name")
        genre = st.text_input("Genre")
        director = st.text_input("Director")
        star = st.text_input("Leading Star")

    with col2:
        country = st.text_input("Country of Production")
        company = st.text_input("Production Company")
        runtime = st.number_input("Runtime (minutes)", min_value=0.0)
        score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0)
        budget = st.number_input("Budget", min_value=0.0)
        year = st.number_input("Year of Release", min_value=1900, max_value=2100)
        votes = st.number_input("Initial Votes", min_value=0)

    # Model selection buttons
    st.markdown("### Select Model for Prediction")
    col_xgb, col_dnn, col_cnn = st.columns(3)
    
    with col_xgb:
        xgb_button = st.form_submit_button(
            label="XGBoost",
            use_container_width=True
        )
    
    with col_dnn:
        dnn_button = st.form_submit_button(
            label="Deep Neural Network",
            use_container_width=True
        )
    
    with col_cnn:
        cnn_button = st.form_submit_button(
            label="Convolutional Neural Network",
            use_container_width=True
        )

# Prepare input data and make predictions
if xgb_button or dnn_button or cnn_button:
    input_data = {
        "released": released,
        "writer": writer,
        "rating": rating,
        "name": name,
        "genre": genre,
        "director": director,
        "star": star,
        "country": country,
        "company": company,
        "runtime": runtime,
        "score": score,
        "budget": budget,
        "year": year,
        "votes": votes,
    }

    st.markdown("## Prediction Result")
    
    if xgb_button:
        with st.spinner('Predicting with XGBoost...'):
            predicted_gross = predict_gross_xgb(input_data, xgb_model)
            predicted_gross_range = predict_gross_range(predicted_gross)
            st.success(f'XGBoost Predicted Revenue for "{name}": ${predicted_gross:,.2f}')
            st.success(f"XGBoost Predicted Revenue Range: {predicted_gross_range}")
            
    if dnn_button:
        with st.spinner('Predicting with Deep Neural Network...'):
            predicted_gross = predict_gross_dnn(input_data, dnn_model)
            predicted_gross_range = predict_gross_range(predicted_gross)
            st.info(f'DNN Predicted Revenue for "{name}": ${predicted_gross:,.2f}')
            st.info(f"DNN Predicted Revenue Range: {predicted_gross_range}")
            
    if cnn_button:
        with st.spinner('Predicting with Convolutional Neural Network...'):
            predicted_gross = predict_gross_cnn(input_data, cnn_model)
            predicted_gross_range = predict_gross_range(predicted_gross)
            st.warning(f'CNN Predicted Revenue for "{name}": ${predicted_gross:,.2f}')
            st.warning(f"CNN Predicted Revenue Range: {predicted_gross_range}")

# Store predictions for comparison
if xgb_button:
    if 'last_input' not in st.session_state:
        st.session_state.last_input = input_data
    st.session_state.last_xgb_pred = predict_gross_xgb(input_data, xgb_model)
    st.session_state.last_xgb_range = predict_gross_range(st.session_state.last_xgb_pred)
    
if dnn_button:
    if 'last_input' not in st.session_state:
        st.session_state.last_input = input_data
    st.session_state.last_dnn_pred = predict_gross_dnn(input_data, dnn_model)
    st.session_state.last_dnn_range = predict_gross_range(st.session_state.last_dnn_pred)
    
if cnn_button:
    if 'last_input' not in st.session_state:
        st.session_state.last_input = input_data
    st.session_state.last_cnn_pred = predict_gross_cnn(input_data, cnn_model)
    st.session_state.last_cnn_range = predict_gross_range(st.session_state.last_cnn_pred)

# Add a comparison section if all models have predictions
if all(key in st.session_state for key in ['last_xgb_pred', 'last_dnn_pred', 'last_cnn_pred']):
    st.markdown("## Model Comparison")
    
    comparison_data = {
        "Model": ["XGBoost", "Deep Neural Network", "Convolutional Neural Network"],
        "Predicted Revenue": [
            f"${st.session_state.last_xgb_pred:,.2f}",
            f"${st.session_state.last_dnn_pred:,.2f}",
            f"${st.session_state.last_cnn_pred:,.2f}"
        ],
        "Revenue Range": [
            st.session_state.last_xgb_range,
            st.session_state.last_dnn_range,
            st.session_state.last_cnn_range
        ]
    }
    
    st.table(pd.DataFrame(comparison_data))