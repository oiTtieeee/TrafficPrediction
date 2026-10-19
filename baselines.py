"""
Classical Machine Learning and Deep Learning Baselines (Linear Regression, Random Forest, LSTM).
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_linear_regression(X_train, y_train):
    """
    Fits an Ordinary Least Squares Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=50, max_depth=15, n_jobs=-1, random_state=42):
    """
    Fits a Random Forest Regressor on flattened lag features.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def build_lstm_model(seq_length=12, hidden_units=64, output_dim=1):
    """
    Constructs a Keras Sequential LSTM neural network.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        model = Sequential([
            LSTM(hidden_units, input_shape=(seq_length, output_dim)),
            Dense(output_dim)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    except ImportError:
        raise ImportError("TensorFlow is required for building LSTM models. Please run: pip install tensorflow")
