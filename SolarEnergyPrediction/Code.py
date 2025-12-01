import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import io
import base64
import webbrowser
import os
import shutil

file_path = 'SolarData.xlsx' 

print("Loading data...")
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"File not found at: {os.path.abspath(file_path)}")
    exit()

target_column = 'generated_power_kw'
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data loaded. Training features: {X_train.shape[1]}")

# Random forest

print("\nTuning Random Forest")

rf_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=67),
    param_distributions=rf_grid,
    n_iter=20,
    cv=2,
    verbose=1,
    n_jobs=-1,
    random_state=67
)

rf_search.fit(X_train, y_train)

best_rf_model = rf_search.best_estimator_
rf_pred = best_rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_best_params = rf_search.best_params_

#cnn

print("\nTuning Deep Learning")

def build_model(hp):
    model = Sequential()
    
    # Layer 1
    model.add(Dense(units=hp.Int('units_1', 32, 256, step=32), 
                    activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    # Layer 2
    model.add(Dense(units=hp.Int('units_2', 32, 128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    
    # Layer 3
    model.add(Dense(units=hp.Int('units_3', 32, 128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_3', 0.1, 0.5, step=0.1)))
    
    model.add(Dense(1, activation='linear'))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
    return model

if os.path.exists('solar_tuner_dir'):
    shutil.rmtree('solar_tuner_dir')

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='solar_tuner_dir',
    project_name='solar_tuning'
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

tuner.search(X_train_scaled, y_train, 
             epochs=200, 
             validation_split=0.2, 
             callbacks=[early_stop],
             verbose=1)

best_dl_model = tuner.get_best_models(num_models=1)[0]
dl_pred = best_dl_model.predict(X_test_scaled).flatten()
dl_pred = np.maximum(dl_pred, 0)

dl_rmse = np.sqrt(mean_squared_error(y_test, dl_pred))
dl_mae = mean_absolute_error(y_test, dl_pred)
dl_r2 = r2_score(y_test, dl_pred)
dl_best_params = tuner.get_best_hyperparameters(num_trials=1)[0].values

#ensemble

print("\nCalculating Ensemble")

ensemble_pred = (rf_pred + dl_pred) / 2

ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ens_mae = mean_absolute_error(y_test, ensemble_pred)
ens_r2 = r2_score(y_test, ensemble_pred)

#image generation

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

#plotting

fig1 = plt.figure(figsize=(18, 6))

# RF Plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, rf_pred, alpha=0.5, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Random Forest\nR2: {rf_r2:.3f} | MAE: {rf_mae:.2f}')

# DL Plot
plt.subplot(1, 3, 2)
plt.scatter(y_test, dl_pred, alpha=0.5, color='#2ecc71')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Deep Learning\nR2: {dl_r2:.3f} | MAE: {dl_mae:.2f}')

# Ensemble Plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, ensemble_pred, alpha=0.5, color='#9b59b6')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Ensemble (Combined)\nR2: {ens_r2:.3f} | MAE: {ens_mae:.2f}')

plt.tight_layout()
plot1_b64 = plot_to_base64(fig1)

#html

# find winner (comparing R2 scores)
scores = {'Random Forest': rf_r2, 'Deep Learning': dl_r2, 'Ensemble': ens_r2}
winner = max(scores, key=scores.get)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Solar Prediction Report</title>
    <style>
        body {{ font-family: Helvetica, Arial, sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
        h1, h2 {{ color: #000; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        .results-table th, .results-table td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .params-box {{ background: #f9f9f9; padding: 15px; border-left: 5px solid #333; margin-bottom: 10px; font-family: monospace; }}
        .param-legend {{ font-size: 0.85em; color: #666; margin-bottom: 20px; font-style: italic; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin-bottom: 20px; }}
        .winner {{ color: #27ae60; font-weight: bold; font-size: 1.2em; }}
        
        .dictionary-section {{ margin-top: 50px; padding-top: 20px; border-top: 2px solid #333; }}
        .term {{ font-weight: bold; font-size: 1.1em; margin-top: 15px; display: block; }}
        .definition {{ margin-left: 0; margin-bottom: 10px; color: #555; }}
    </style>
</head>
<body>
    <h1>Solar Energy Output Prediction</h1>
    <p>Winner: <span class="winner">{winner}</span></p>

    <h2>Model Performance</h2>
    <table class="results-table">
        <tr>
            <th>Model</th>
            <th>R2 Score</th>
            <th>MAE (Avg Error kW)</th>
            <th>RMSE (Peak Error kW)</th>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>{rf_r2:.4f}</td>
            <td>{rf_mae:.4f}</td>
            <td>{rf_rmse:.4f}</td>
        </tr>
        <tr>
            <td>Deep Learning (Neural Network)</td>
            <td>{dl_r2:.4f}</td>
            <td>{dl_mae:.4f}</td>
            <td>{dl_rmse:.4f}</td>
        </tr>
        <tr style="background-color: #e8f8f5; font-weight: bold;">
            <td>Ensemble (Combined)</td>
            <td>{ens_r2:.4f}</td>
            <td>{ens_mae:.4f}</td>
            <td>{ens_rmse:.4f}</td>
        </tr>
    </table>

    <h2>Best Hyperparameters</h2>
    <p><strong>Random Forest Settings:</strong></p>
    <div class="params-box">{rf_best_params}</div>
    
    <p><strong>Neural Network Settings:</strong></p>
    <div class="params-box">{dl_best_params}</div>
    <div class="param-legend">
        <b>units:</b> Neurons <br>
        <b>dropout:</b> Percentage of neurons turned off during training to prevent memorization.
    </div>

    <h2>Visualizations</h2>
    <img src="data:image/png;base64,{plot1_b64}" alt="Prediction Comparison">

    <div class="dictionary-section">
        <h2>Term Dictionary</h2>

        <span class="term">Ensemble</span>
        <div class="definition">
            A technique that combines the predictions of multiple models (in this case, averaging the Tree and the Neural Network). 
            It creates a "Super Prediction" that is usually more stable because the errors of one model typically cancel out the errors of the other.
        </div>

        <span class="term">MAE (Mean Absolute Error)</span>
        <div class="definition">
            The average size of the error, treating all errors equally. 
            Unlike RMSE (which squares errors and punishes outliers heavily), MAE is better for solar data because it isn't thrown off by sudden, random cloud movements.
        </div>
        
        <span class="term">Hyperparameter Searching</span>
        <div class="definition">
            The process of automating the "guessing game" of machine learning. The computer tests many variations to find the optimal configuration.
        </div>

        <span class="term">R2 Score</span>
        <div class="definition">
            Accuracy score where 1.0 is perfect and 0.0 is poor.
        </div>
    </div>
</body>
</html>
"""

output_file = "solar_ensemble_results.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Success! Results saved to {output_file}")
webbrowser.open('file://' + os.path.realpath(output_file))