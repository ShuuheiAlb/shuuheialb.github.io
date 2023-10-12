import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



# Time Series Analysis: LTSM
# Select the relevant columns for modeling
data = pd.read_csv("extracted.csv")
selected_columns = ["Energy", "Mean Temperature"]
data["Mean Temperature"].interpolate(method="linear", inplace=True) # Simple interpol

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[selected_columns])

# Set the sequence length and generate sequences
sequence_length = 7  # Number of past observations to consider
generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=32)

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(generator[0][0], generator[0][1], test_size=0.2, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, len(selected_columns))))
model.add(Dense(len(selected_columns)))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_X, train_y, epochs=10, batch_size=32)
from sklearn.model_selection import train_test_split

# Evaluate the model as needed