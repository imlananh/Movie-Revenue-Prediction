import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from feature_scaling import prepare_features

# Load dataset
df = pd.read_csv("revised datasets/output.csv")

# Preprocess and scale data
X, y = prepare_features(df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build DNN model
model = Sequential([
    Dense(128, input_dim=X.shape[1]),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(32),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),

    Dense(1)
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mse')

# Callbacks: EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100
    return r2, mse, msle, mape

train_r2, train_mse, train_msle, train_mape = calculate_metrics(
    y_train, train_predictions.flatten()
)
test_r2, test_mse, test_msle, test_mape = calculate_metrics(
    y_test, test_predictions.flatten()
)

print(f"\nTraining Metrics:")
print(f"R2 score: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"MLSE: {train_msle:.4f}")
print(f"MAPE: {train_mape:.2f}%")

print(f"\nTest Metrics:")
print(f"R2 score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MSLE: {test_msle:.4f}")
print(f"MAPE: {test_mape:.2f}%")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color="blue", label="Train")
plt.scatter(y_test, test_predictions, color="red", label="Test")
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
