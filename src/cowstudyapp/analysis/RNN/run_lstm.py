from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from cowstudyapp.config import ConfigManager, LSTMConfig, AnalysisConfig
from cowstudyapp.utils import from_posix, from_posix_col


class Cow_Data:

    def __init__(self, device_id:int, config: ConfigManager):
        self.device_id = device_id
        self.config = config

        self.data_map: [datetime.date, pd.DataFrame] = {}

    def add_day_data(self,date, data):
        self.data_map[date] = data #[data[self.config.analysis.lstm.features].to_numpy(), data[['activity']].to_numpy()]

    def get_ordered_X(self):

        ordered_dates = sorted([date for date in self.data_map.keys()])
        X_list = []

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        for date in ordered_dates: 
            date_data = self.data_map[date].sort_index()

            if len(date_data) == 288:
                X_list.append(date_data[self.config.analysis.lstm.features])


            else:
                print(f"ERROR ON device {self.device_id}, {date}, there are only {len(date_data)} records.")

            # print(date_data.head())





    


class LSTM_Model:

    def __init__(self, config: ConfigManager):
        self.config = config


        tf.random.set_seed(self.config.analysis.random_seed)
        np.random.seed(self.config.analysis.random_seed)


    def _get_target_dataset(self, add_step_and_angle=True):
        df = pd.read_csv(self.config.analysis.target_dataset)
        step_size = self.config.analysis.gps_sample_interval//60
        dfs = []
        
        for device_id, cow_data in df.groupby("device_id"):
            if add_step_and_angle:
                cow_data = cow_data.sort_values('posix_time')
                
                # Calculate time gaps and interpolate as before
                time_gaps = cow_data['posix_time'].diff()
                double_gaps = time_gaps == 2 * step_size * 60
                if double_gaps.any():
                    cow_data = cow_data.interpolate(method='linear', limit_area='inside')
                
                # Calculate forward-looking step (distance to next point)
                # Use shift(-1) to look forward
                cow_data['step'] = np.sqrt(
                    (cow_data['utm_easting'].shift(-1) - cow_data['utm_easting'])**2 +
                    (cow_data['utm_northing'].shift(-1) - cow_data['utm_northing'])**2
                )
                
                # Calculate vectors (forward-looking displacement vectors)
                x_diff_current = cow_data['utm_easting'].shift(-1) - cow_data['utm_easting']  # Current vector x
                y_diff_current = cow_data['utm_northing'].shift(-1) - cow_data['utm_northing']  # Current vector y
                
                x_diff_prev = cow_data['utm_easting'] - cow_data['utm_easting'].shift(1)  # Previous vector x
                y_diff_prev = cow_data['utm_northing'] - cow_data['utm_northing'].shift(1)  # Previous vector y
                
                # Calculate headings
                current_heading = np.arctan2(y_diff_current, x_diff_current)  # Angle of current vector
                prev_heading = np.arctan2(y_diff_prev, x_diff_prev)  # Angle of previous vector
                
                # Calculate the difference in heading (φ)
                cow_data['angle'] = current_heading - prev_heading

                # print(cow_data[['posix_time', 'device_id', 'utm_easting', 'utm_northing', 'step', 'angle']].head(5)) #'prev_heading', 'current_heading',
                # print(cow_data[['posix_time', 'device_id', 'utm_easting', 'utm_northing', 'step', 'angle']].tail(5)) #'prev_heading', 'current_heading',
                # return
            
            time_range = pd.date_range(
                start=pd.to_datetime(cow_data['posix_time'].min(), unit='s'),
                end=pd.to_datetime(cow_data['posix_time'].max(), unit='s'),
                freq=f'{step_size}min'
            )

            new_df = pd.DataFrame({'posix_time': time_range.astype('int64')//1e9, 'device_id': device_id})
            dfs.append(pd.merge(new_df, cow_data, on=['posix_time', 'device_id'], how='left'))


           
        df_out = pd.concat(dfs).sort_values(['device_id', 'posix_time'])


        if 'mt' not in df.columns:
            df_out['mt'] = from_posix_col(df_out['posix_time'], self.config.analysis.timezone)
        
        # Extract date component
        df_out['date'] = df_out['mt'].dt.date

        return df_out
    
    def normalize_features(self, df, features):
        """Normalize features before sequence building"""
        
        scaler = StandardScaler()
        
        # Normalize features
        normalized_data = scaler.fit_transform(df[features])
        df[features] = normalized_data

        # Store scaler
        self.scaler = scaler
        return df


    def do_loocv(self):
        df = self._get_target_dataset()

        required_cols: List[str] = self.config.analysis.lstm.features + ["posix_time", "date", 'device_id', 'activity']
        print(required_cols)
        df = self.normalize_features(df, self.config.analysis.lstm.features)

        df = df[required_cols].copy()

        print(df.columns)


        cows = self._build_sequences_opo(df)


        predictions = []
        actual_states = []



        for training_idx, training_cow in enumerate(cows):
            training_cow.get_ordered_X()
            # training_sequences = np.array(training_day_dict.items())

            # print(training_sequences)

            # return
            


    def _build_sequences_opo(self, df):
        max_length = self.config.analysis.lstm.max_length
        features = self.config.analysis.lstm.features
        step_size = self.config.analysis.gps_sample_interval//60  # Sample interval in minutes
        
        if max_length == 'daily':
            X = []
            y = []
            device_date_map = []
            
            for device_id, device_data in df.groupby("device_id"):
                for date, day_data in device_data.groupby("date"):

                    
                    # Get expected length for a full day
                    minutes_in_day = 24 * 60
                    expected_records = minutes_in_day // step_size
                    
                    if len(df) < expected_records * 0.5:  # Skip days with too few records
                        print(f"Skipping day with insufficient data: {date}, {len(df)}/{expected_records}")
                        continue
                    
                    elif len(df) == int(expected_records - (60//step_size)):
                        print(f"Missing data due to timezone shift. Inserting {(60//step_size)} values to the end of the sequence.")
                        df

                    # Extract features, replace NaN with 0 (will be masked)
                    seq = df[features].fillna(0).values
                    
                    # Add sequence to our dataset
                    X.append(seq)
                    
                    # Extract target (activity)
                    activity_idx = features.index('activity') if 'activity' in features else None
                    if activity_idx is not None:
                        # Use the last valid activity or a default
                        last_valid_activity = merged_day['activity'].dropna().iloc[-1] if not merged_day['activity'].dropna().empty else -1
                        y.append(last_valid_activity)
                    else:
                        y.append(-1)
                        
                    device_date_map.append((device_id, date))
            
            X = np.array(X)
            y = np.array(y)
            
            # Ensure your model has a Masking layer at the beginning
            # model = tf.keras.Sequential([
            #     tf.keras.layers.Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
            #     # rest of your model...
            # ])
            
            return {
                'X': X,
                'y': y,
                'device_date_map': device_date_map
            }




    # def build_many_to_one_model(self, train_x, train_y, test_x, test_y):
    #     """Build RNN model that predicts single activity from sequence"""
    #     sequence_length = self.config.analysis.lstm.max_length
    #     n_features = len(self.config.analysis.lstm.features)
    #     n_classes = len(self.config.analysis.lstm.states)


    #     print("\nClass distribution in training data:")
    #     train_series = pd.Series(train_y).map(self.config.analysis.inv_activity_map)
    #     print(train_series.value_counts())

    #     print(np.array(range(n_classes)))
    #     # Calculate class weights using numeric labels
    #     n_classes = len(self.config.analysis.activity_labels)
    #     class_weights = compute_class_weight(
    #         class_weight='balanced',
    #         classes=np.array(range(n_classes)),
    #         y=train_y
    #     )

    #     # # Calculate class weights using integer labels directly
    #     # class_weights = compute_class_weight(
    #     #     class_weight='balanced',
    #     #     classes=np.arange(n_classes),
    #     #     y=train_y
    #     # )
    #     class_weight_dict = dict(zip(range(n_classes), class_weights))


    #     for i, weight in class_weight_dict.items():
    #         print(f"{self.config.analysis.activity_labels[i]}: {weight:.3f}")


    #     model = Sequential([
    #         Masking(mask_value=0., input_shape=(sequence_length, n_features)),
            
    #         LSTM(256, 
    #             return_sequences=True,
    #             activation='tanh',              # Default activation
    #             recurrent_activation='sigmoid', # Gate activation
    #             use_bias=True,
    #             unit_forget_bias=True,          # Initialize forget gate bias to 1
    #             recurrent_dropout=0.2,          # Dropout for recurrent connections
    #             dropout=0.3,                    # Dropout for inputs
    #             kernel_regularizer=tf.keras.regularizers.L2(0.01),
    #             recurrent_regularizer=tf.keras.regularizers.L2(0.01)
    #             ),          
            
    #         LSTM(128, 
    #             return_sequences=False,
    #             recurrent_dropout=0.2,
    #             dropout=0.3,
    #             kernel_regularizer=tf.keras.regularizers.L2(0.01),
    #             recurrent_regularizer=tf.keras.regularizers.L2(0.01)
    #             ),          # Dropout for inputs
            
    #         Dense(128, activation='relu'),
    #         Dropout(0.3),
    #         Dense(n_classes, activation='softmax')
    #     ])

    #     # model = Sequential([
    #     #     Masking(mask_value=0., input_shape=(sequence_length, n_features)),
    #     #     SimpleRNN(256, return_sequences=True),
    #     #     Dropout(0.3),
            
    #     #     SimpleRNN(128, return_sequences=False), #, activation='relu'
    #     #     Dropout(0.3),
            
    #     #     Dense(128, activation='relu'),
    #     #     Dropout(0.3),
    #     #     # Dense(16, activation='relu'),
    #     #     Dense(n_classes, activation='softmax')
    #     # ])




    #     # learning rate schedule
    #     initial_learning_rate = 0.001
    #     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate,
    #         decay_steps=1000,
    #         decay_rate=0.9
    #     )


    #     optimizer = tf.keras.optimizers.Adam(
    #         learning_rate=lr_schedule,
    #         clipnorm=1.0,  # Clip gradients to prevent explosions
    #         clipvalue=0.5
    #     )
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)


    #     model.compile(
    #         optimizer=optimizer,
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy']
    #     )
        
    #     history = model.fit(
    #         train_x, 
    #         train_y,
    #         validation_split=0.2,
    #         epochs=self.config.analysis.epochs,  # Increase epochs
    #         batch_size=16,
    #         class_weight=class_weight_dict,
    #         callbacks=[
    #             tf.keras.callbacks.EarlyStopping(
    #                 monitor='val_loss',
    #                 patience=15,          # Let it train longer
    #                 restore_best_weights=True,
    #                 min_delta=0.0005     # More sensitive to improvements
    #             ),
    #             tf.keras.callbacks.ReduceLROnPlateau(
    #                 monitor='val_loss',
    #                 factor=0.2,          # More aggressive reduction
    #                 patience=7,
    #                 min_lr=0.00001
    #             )
    #         ]
    #     )        
    #     # Predict and evaluate
    #     pred_y = model.predict(test_x, verbose=0)  # Suppress prediction progress bar
    #     pred_y_classes = np.argmax(pred_y, axis=1)
    #     # pred_y_labels = le.inverse_transform(pred_y_classes)
       

    #     return model, pred_y_classes, history




    # def plot_confusion_matrix(self, y_true, y_pred):
    #     """Plot confusion matrix of predictions using numeric labels"""
    #     cm = confusion_matrix(y_true, y_pred, labels=range(len(self.config.analysis.activity_labels)))
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(
    #         cm, 
    #         annot=True, 
    #         fmt='d', 
    #         xticklabels=self.config.analysis.activity_labels,  # Use string labels for display
    #         yticklabels=self.config.analysis.activity_labels
    #     )
    #     plt.title('Confusion Matrix')
    #     plt.ylabel('True Label')
    #     plt.xlabel('Predicted Label')
    #     plt.show()

    # def plot_training_histories(self, histories):
    #     """Plot training histories across all folds with different lengths"""
    #     plt.figure(figsize=(12, 4))
        
    #     # Find max length of histories
    #     max_epochs = max(len(h.history['loss']) for h in histories)
        
    #     # Initialize arrays for mean and std calculations
    #     losses = np.zeros((len(histories), max_epochs))
    #     accuracies = np.zeros((len(histories), max_epochs))
        
    #     # Fill arrays with padding
    #     for i, h in enumerate(histories):
    #         # Pad or truncate loss
    #         curr_len = len(h.history['loss'])
    #         losses[i, :curr_len] = h.history['loss']
    #         losses[i, curr_len:] = h.history['loss'][-1]  # Pad with last value
            
    #         # Pad or truncate accuracy
    #         accuracies[i, :curr_len] = h.history['accuracy']
    #         accuracies[i, curr_len:] = h.history['accuracy'][-1]  # Pad with last value
        
    #     # Plot Loss
    #     plt.subplot(1, 2, 1)
    #     epochs = range(1, max_epochs + 1)
        
    #     # Plot individual histories
    #     for i in range(len(histories)):
    #         plt.plot(epochs[:len(histories[i].history['loss'])], 
    #                 histories[i].history['loss'], 
    #                 'b-', alpha=0.1)
        
    #     # Plot mean and std
    #     mean_loss = np.mean(losses, axis=0)
    #     std_loss = np.std(losses, axis=0)
    #     plt.plot(epochs, mean_loss, 'r-', label='Mean Loss')
    #     plt.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss, 
    #                     color='r', alpha=0.2, label='±1 std')
    #     plt.title('Training Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
        
    #     # Plot Accuracy
    #     plt.subplot(1, 2, 2)
        
    #     # Plot individual histories
    #     for i in range(len(histories)):
    #         plt.plot(epochs[:len(histories[i].history['accuracy'])], 
    #                 histories[i].history['accuracy'], 
    #                 'b-', alpha=0.1)
        
    #     # Plot mean and std
    #     mean_acc = np.mean(accuracies, axis=0)
    #     std_acc = np.std(accuracies, axis=0)
    #     plt.plot(epochs, mean_acc, 'r-', label='Mean Accuracy')
    #     plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, 
    #                     color='r', alpha=0.2, label='±1 std')
    #     plt.title('Training Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
        
    #     plt.tight_layout()
    #     plt.show()

    #     # Print final metrics
    #     print("\nTraining Summary:")
    #     print(f"Final Mean Loss: {mean_loss[-1]:.4f} ± {std_loss[-1]:.4f}")
    #     print(f"Final Mean Accuracy: {mean_acc[-1]:.4f} ± {std_acc[-1]:.4f}")
