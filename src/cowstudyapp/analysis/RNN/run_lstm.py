from datetime import datetime
from pathlib import Path
import logging
# from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, Masking

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Input, BatchNormalization, Conv1D


from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from cowstudyapp.config import ConfigManager, LSTMConfig, AnalysisConfig
from cowstudyapp.utils import from_posix, from_posix_col





    


class LSTM_Model:

    def __init__(self, config: ConfigManager):
        self.config = config

        self.activity_map = {"Grazing": 0, "Resting":1, "Traveling": 2, np.nan:-1}
        self.inv_activity_map = {self.activity_map[k]: k for k in self.activity_map.keys()}


        tf.random.set_seed(self.config.analysis.random_seed)
        np.random.seed(self.config.analysis.random_seed)
        self.masking_val = -9999


    def _get_target_dataset(self, add_step_and_angle=True):
        df = pd.read_csv(self.config.analysis.target_dataset)
        step_size = self.config.analysis.gps_sample_interval//60
        dfs = []
        
        for device_id, cow_data in df.groupby("device_id"):
            # Sort data first
            cow_data = cow_data.sort_values('posix_time')
            
            # Create full index
            time_range = pd.date_range(
                start=pd.to_datetime(cow_data['posix_time'].min(), unit='s'),
                end=pd.to_datetime(cow_data['posix_time'].max(), unit='s'),
                freq=f'{step_size}min'
            )
            full_index_df = pd.DataFrame({
                'posix_time': time_range.astype('int64')//1e9, 
                'device_id': device_id
            })
            
            # Merge to create a complete time series with potential NaN values
            cow_data = pd.merge(full_index_df, cow_data, on=['posix_time', 'device_id'], how='left')
            
            # Now apply the isolated NaN interpolation
            numeric_cols = cow_data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                # Skip device_id and posix_time
                if col in ['device_id', 'posix_time']:
                    continue
                    
                # Find isolated NaN values (valid values on both sides)
                series = cow_data[col].copy()
                mask = series.isna() & series.shift(1).notna() & series.shift(-1).notna()
                
                # Only interpolate where mask is True (isolated NaNs)
                if mask.any():
                    # Create a temporary series with only the isolated NaNs for interpolation
                    temp = series.copy()
                    temp[~mask] = temp[~mask].ffill()  # Fill non-isolated NaNs temporarily
                    temp = temp.interpolate(method='linear')
                    
                    # Apply only the interpolated isolated NaNs back to original series
                    series[mask] = temp[mask]
                    cow_data[col] = series
            
            if add_step_and_angle:
                # Calculate forward-looking step (distance to next point)
                cow_data['step'] = np.sqrt(
                    (cow_data['utm_easting'].shift(-1) - cow_data['utm_easting'])**2 +
                    (cow_data['utm_northing'].shift(-1) - cow_data['utm_northing'])**2
                )
                
                # Calculate vectors (forward-looking displacement vectors)
                x_diff_current = cow_data['utm_easting'].shift(-1) - cow_data['utm_easting']
                y_diff_current = cow_data['utm_northing'].shift(-1) - cow_data['utm_northing']
                x_diff_prev = cow_data['utm_easting'] - cow_data['utm_easting'].shift(1)
                y_diff_prev = cow_data['utm_northing'] - cow_data['utm_northing'].shift(1)
                
                # Calculate headings
                current_heading = np.arctan2(y_diff_current, x_diff_current)
                prev_heading = np.arctan2(y_diff_prev, x_diff_prev)
                
                # Calculate the difference in heading (φ)
                cow_data['angle'] = current_heading - prev_heading
            
            # Add to the list of processed dataframes
            dfs.append(cow_data)
        
        # Combine all device data
        df_out = pd.concat(dfs).sort_values(['device_id', 'posix_time'])
        
        # Add time columns
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


    def _build_sequences_opo(self, df:pd.DataFrame):
        max_length = self.config.analysis.lstm.max_length
        features = self.config.analysis.lstm.features
        step_size = self.config.analysis.gps_sample_interval//60  # Sample interval in minutes

        minutes_in_day = 24 * 60
        expected_records = minutes_in_day // step_size
        self.sequence_length = expected_records

        lost_hour = (60//step_size)
        if max_length == 'daily':
            Cow_Date_Key = []
            X = []
            Y = []
            for device_id, device_data in df.groupby("device_id"):
                # self.data_map[device_id] = {}
                for date, day_data in device_data.groupby("date"):
                    if len(day_data) < expected_records * 0.5:  # Skip days with too few records
                        # print(f"Skipping day with insufficient data: {date}, {len(day_data)}/{expected_records}")
                        continue
                    
                    # Extract features, replace NaN with masking value
                    seq :pd.DataFrame = day_data[features].copy().reset_index(drop=True) #.fillna(self.masking_val)
                    labels :pd.DataFrame = day_data[['activity']].copy().reset_index(drop=True)

                    labels = labels.map(lambda x: self.activity_map.get(x, -1))

                    if len(seq) == expected_records - lost_hour:
                        # print(f"Missing data due to timezone shift. Inserting {lost_hour} values to the end of the sequence.")
                        masking_df = pd.DataFrame({
                            col: [np.nan] * lost_hour for col in seq.columns
                        })
                        seq = pd.concat([seq, masking_df], ignore_index=True)

                        nan_labels = pd.DataFrame({'activity': [np.nan] * lost_hour})
                        nan_labels = nan_labels.map(lambda x: self.activity_map.get(x, -1))
                        labels = pd.concat([labels, nan_labels], ignore_index=True)

                    else:
                        pass
                    seq.fillna(self.masking_val, inplace=True)
                    labels.fillna(self.activity_map[np.nan],inplace=True)
                    
                    # Add sequence to our dataset
                    Cow_Date_Key.append([device_id,date])
                    X.append(seq)
                    Y.append(labels)
                    
            Cow_Date_Key = np.array(Cow_Date_Key)
            X = np.array(X, dtype=np.float32)  # Ensure float32 for X
            Y = np.array(Y, dtype=np.int32)    # Ensure int32 for Y

            return {
                'Cow_Date_Key': Cow_Date_Key,
                'X': X,
                'Y': Y,
            }

        # else:


    def dont_do_loocv(self):



        df = self._get_target_dataset()
        
        # Enable mixed precision to speed up computations
        if tf.config.list_physical_devices('GPU'):
            print("Using GPU with mixed precision")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        else:
            # Attempt to optimize CPU operations
            try:
                # Set number of threads for parallel processing
                tf.config.threading.set_intra_op_parallelism_threads(8)
                tf.config.threading.set_inter_op_parallelism_threads(8)
            except:
                print("Could not set parallelism threads.")
        
        

        required_cols: List[str] = self.config.analysis.lstm.features + ["posix_time", "date", 'device_id', 'activity']
        print(required_cols)
        df = self.normalize_features(df, self.config.analysis.lstm.features)
        df = df[required_cols].copy()
        
        print(df.columns)
        self.nfeatures = len(self.config.analysis.lstm.features)
        self.nclasses = len(self.config.analysis.lstm.states)
        
        sequences = self._build_sequences_opo(df)
        
        Cow_Date_Key = sequences['Cow_Date_Key']
        X = sequences['X']
        Y = sequences['Y'].squeeze(axis=2)
        






        val_split = 0.2

        sequences = self._build_sequences_opo(df)
        Cow_Date_Key = sequences['Cow_Date_Key']
        X = sequences['X']
        Y = sequences['Y'].squeeze(axis=2)
        
        # Calculate label density for each sequence
        label_density = []
        for i in range(len(Y)):
            # Calculate percentage of timesteps that have valid labels
            valid_labels = (Y[i] != self.activity_map[np.nan])
            density = np.mean(valid_labels)
            label_density.append(density)
        
        # Only consider sequences with sufficient labels for validation
        min_label_density = 0.1  # At least 10% of timesteps must have labels
        valid_indices = np.where(np.array(label_density) >= min_label_density)[0]
        
        if len(valid_indices) < 10:
            print("WARNING: Very few sequences with sufficient labels found!")
            # Fallback to using all sequences
            valid_indices = np.arange(len(X))
        
        # Now split only among valid indices
        np.random.shuffle(valid_indices)
        split_idx = int(len(valid_indices) * (1 - val_split))
        
        train_indices = valid_indices[:split_idx]
        val_indices = valid_indices[split_idx:]
        
        train_X, val_X = X[train_indices], X[val_indices]
        train_Y, val_Y = Y[train_indices], Y[val_indices]
        
        # Print label statistics
        train_label_count = np.sum(train_Y != self.activity_map[np.nan])
        val_label_count = np.sum(val_Y != self.activity_map[np.nan])
        print(f"Training set: {len(train_X)} sequences with {train_label_count} labeled timesteps")
        print(f"Validation set: {len(val_X)} sequences with {val_label_count} labeled timesteps")









        # Create output directory for models
        output_dir = Path(self.config.analysis.output_dir) / "lstm_model"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Train a single global model on all data
        print("\n==== Training Single Global Model ====")
        
        # # Use a small validation split

        # indices = np.arange(len(X))
        # np.random.shuffle(indices)
        # split_idx = int(len(indices) * (1 - val_split))
        
        # train_indices = indices[:split_idx]
        # val_indices = indices[split_idx:]
        
        # train_X, val_X = X[train_indices], X[val_indices]
        # train_Y, val_Y = Y[train_indices], Y[val_indices]
        
        # # Train single model (no test data provided)
        # model, history, _, _, _ = self._make_LSTM(
        #     train_X, train_Y, val_X=val_X, val_Y=val_Y, test_X=None, test_Y=None
        # )
        
        # Train single model (no test data provided)
        model, history, _, _, _ = self._make_LSTM(
            X, Y, val_X=None, val_Y=None, test_X=None, test_Y=None
        )
        
        # Save the global model
        model_path = output_dir / "global_lstm_model.keras"
        model.save(model_path)
        print(f"Global model saved to {model_path}")
        
        # Plot training history
        self._plot_single_history(history)
        
        return model, history
    



    def do_loocv(self):
        df = self._get_target_dataset()
        
        required_cols: List[str] = self.config.analysis.lstm.features + ["posix_time", "date", 'device_id', 'activity']
        print(required_cols)
        df = self.normalize_features(df, self.config.analysis.lstm.features)
        df = df[required_cols].copy()
        
        print(df.columns)
        self.nfeatures = len(self.config.analysis.lstm.features)
        self.nclasses = len(self.config.analysis.lstm.states)
        
        sequences = self._build_sequences_opo(df)
        
        Cow_Date_Key = sequences['Cow_Date_Key']
        X = sequences['X']
        Y = sequences['Y'].squeeze(axis=2)
        
        # Enable mixed precision to speed up computations
        if tf.config.list_physical_devices('GPU'):
            print("Using GPU with mixed precision")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        else:
            # Attempt to optimize CPU operations
            try:
                # Set number of threads for parallel processing
                tf.config.threading.set_intra_op_parallelism_threads(8)
                tf.config.threading.set_inter_op_parallelism_threads(8)
            except:
                print("Could not set parallelism threads.")
        
        # For storing results
        all_predictions = []
        all_actual = []
        all_models = []
        all_histories = []
        
        # Create output directory for models
        output_dir = Path(self.config.analysis.output_dir) / "lstm_models"
        output_dir.mkdir(exist_ok=True, parents=True)
        




        for i, test_cow in enumerate(df.device_id.unique()):
            print(f"\n==== Training model for test cow {test_cow} ({i+1}/{len(df.device_id.unique())}) ====")
            
            test_idx = Cow_Date_Key[:,0] == test_cow
            train_idx = Cow_Date_Key[:,0] != test_cow
            
            test_X = X[test_idx]
            test_Y = Y[test_idx]
            train_X = X[train_idx]
            train_Y = Y[train_idx]
            
            # Train the model and get results
            model, history, pred_final, test_Y_clean, test_mask = self._make_LSTM(
                test_X, test_Y, train_X, train_Y
            )
            
            # Save model
            model_path = output_dir / f"model_leave_{test_cow}_out.keras"
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Store results
            all_predictions.append(pred_final)
            all_actual.append(test_Y)
            all_models.append(model)
            all_histories.append(history)
            
            # Calculate accuracy only on valid data points (where mask is 1)
            valid_indices = test_mask > 0
            accuracy = np.mean(pred_final[valid_indices] == test_Y_clean[valid_indices])
            print(f"Accuracy on valid data points: {accuracy:.4f}")
            
            # Display class distribution
            unique, counts = np.unique(test_Y_clean[valid_indices], return_counts=True)
            print("Class distribution in test data:")
            for class_idx, count in zip(unique, counts):
                label = self.inv_activity_map.get(class_idx, f"Unknown ({class_idx})")
                print(f"  {label}: {count}")
        
        # Save and display overall results
        self._summarize_results(all_predictions, all_actual, all_histories)
        return all_models, all_histories






    def _make_LSTM(self, X, Y, test_X=None, test_Y=None, val_X=None, val_Y=None):
        """Build and train LSTM model with improved handling of imbalanced data
        
        Parameters:
        -----------
        train_X, train_Y: Training data
        test_X, test_Y: Test data for LOOCV (optional, can be None)
        val_X, val_Y: Validation data (optional, used in global model mode)
        
        Returns:
        --------
        model, history, pred_final, test_Y_clean, test_mask
        """
        # # Handle train/validation split
        # use_external_validation = val_X is not None and val_Y is not None
        # use_test_prediction = test_X is not None and test_Y is not None


        n_epochs = self.config.analysis.lstm.epochs
        # # Create masks for unknown values in training data
        # train_mask = (train_Y != self.activity_map[np.nan]).astype(np.float32)

        # # Debug mask information
        # total_timesteps = train_Y.size
        # masked_timesteps = np.sum(train_mask == 0)
        # print(f"Total training timesteps: {total_timesteps}")
        # print(f"Masked timesteps: {masked_timesteps} ({masked_timesteps/total_timesteps:.2%})")
        

        # # Clean training data for model use
        # train_Y_clean = np.copy(train_Y)
        # train_Y_clean[train_Y_clean == self.activity_map[np.nan]] = 0
        



        # # DEBUGGING 

        # # Calculate class weights only on valid data
        # valid_labels = train_Y[train_mask > 0]
        # class_counts = np.bincount(valid_labels)
        
        # # Ensure all classes are represented
        # if len(class_counts) < self.nclasses:
        #     print("WARNING: Some classes are missing in valid data!")
        #     class_counts = np.pad(class_counts, (0, self.nclasses - len(class_counts)))
        
        # class_counts = np.maximum(class_counts, 1)  # Ensure no zeros
        
        # total = np.sum(class_counts)
        # class_weights = {i: total / (self.nclasses * class_counts[i]) for i in range(self.nclasses)}
        
        # # print(f"Class distribution in valid training data: {class_counts}")
        # # print(f"Class weights: {class_weights}")
        
        # # Create sample weight matrix - verify with explicit examples
        # sample_weights = np.ones_like(train_Y_clean, dtype=np.float32)
        # sample_weights = sample_weights * train_mask  # Mask unknown values
        

        # # Process test data if provided
        # if True:
        #     test_mask = int((test_Y != self.activity_map[np.nan])).astype(np.float32)
        #     test_Y_clean = np.copy(test_Y)
        #     test_Y_clean[test_Y_clean == self.activity_map[np.nan]] = 0
        # else:
        #     # Create dummy values that will be ignored
        #     test_mask = np.array([0])
        #     test_Y_clean = np.array([0])
        
        # # Process validation data if provided
        # if use_external_validation:
        #     val_mask = int((val_Y != self.activity_map[np.nan])).astype(np.float32)
        #     val_Y_clean = np.copy(val_Y)
        #     val_Y_clean[val_Y_clean == self.activity_map[np.nan]] = 0
        #     validation_data = (val_X, val_Y_clean, val_mask)
        #     validation_split = None

        #     valid_val_labels = np.sum(val_mask)
        #     print(f"Validation set: {valid_val_labels} valid labels out of {val_Y.size}")
        # else:
        #     # Use internal validation split
        #     validation_data = None
        #     validation_split = 0.3
        

        # # Calculate class weights to handle imbalance
        # valid_labels = train_Y_clean[train_mask > 0].flatten()
        # class_counts = np.bincount(valid_labels)
        # total = np.sum(class_counts)
        # class_weights = {i: total / (len(class_counts) * count) if count > 0 else 1.0 
        #                 for i, count in enumerate(class_counts)}
        
        # print(f"Class distribution in training data: {class_counts}")
        # print(f"Class weights: {class_weights}")
        
        # # Create a combined weight matrix that includes both masking and class weights
        # sample_weights = np.ones_like(train_Y_clean, dtype=np.float32)
        
        # # First apply the mask (0 weight for unknown values)
        # sample_weights = sample_weights * train_mask
        
        # # Then apply class weights (each timestep gets weight based on its class)
        # for class_id, weight in class_weights.items():
        #     class_mask = (train_Y_clean == class_id)
        #     sample_weights[class_mask] = weight * sample_weights[class_mask]
        


        # assume you’ve already grouped into daily sequences:
        #  X: np.array, shape (batch, seq_len, n_features), filled with your mask_value for missing *features*
        #  Y: np.array, shape (batch, seq_len), dtype float, containing integer labels or np.nan for “no label”

        # Convert labels to int (fill nan→0) and build a 0/1 mask for where a label actually exists
        Y_int   = np.nan_to_num(Y, nan=0).astype('int32')        # any “dummy” class 0
        mask_y  = (~np.isnan(Y)).astype('float32')               # 1.0 where label is valid, 0.0 otherwise



        # Create a simpler, faster model
        model = Sequential([
            Input((self.sequence_length, self.nfeatures)),
            Masking(mask_value=self.masking_val),
            
            # Conv1D layers to capture local patterns
            Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            # Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            # BatchNormalization(),     
            # # Single LSTM layer with more units

            # LSTM(128, 
            #     return_sequences=True,
            #     # activation='tanh',
            #     # recurrent_activation='sigmoid',
            #     dropout=0.2,
            #     # recurrent_dropout=0.2,
            #     ),          
            LSTM(64, 
                return_sequences=True,
                # activation='tanh',
                # recurrent_activation='sigmoid',
                dropout=0.2,
                recurrent_dropout=0.2,
                ),          
            
            # Time distributed layers
            tf.keras.layers.TimeDistributed(Dense(64, activation='relu')),
            tf.keras.layers.TimeDistributed(Dropout(0.2)),
            tf.keras.layers.TimeDistributed(Dense(self.nclasses, activation='softmax'))
        ])
        
        batchsize=16
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=n_epochs * len(train_X) // batchsize,
        #     alpha=0.00001  # Minimum learning rate
        # )
        
    #     # learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )


        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        #     # optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #     metrics=['accuracy'],
        #     sample_weight_mode='temporal'
        # )

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
            # sample_weight_mode='temporal'              # <-- tells Keras that sample_weight is (batch, time)
        )
        
        model.summary()
        # print(f"Train_Y Shape: {train_Y.shape}")
        # print(f"Train_X Shape: {train_X.shape}")
        

        
        # Create a tensorboard callback
        log_dir = Path(self.config.analysis.output_dir) / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1,
            profile_batch=0  # Disable profiling for speed
        )
        

        history = model.fit(
            X, 
            Y_int,
            # validation_data=validation_data,
            # validation_split=validation_split,
            sample_weight=mask_y,
            epochs=n_epochs,
            batch_size=batchsize,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    # monitor='loss',
                    patience=15,
                    restore_best_weights=True
                ),
                # tf.keras.callbacks.ReduceLROnPlateau(
                #     monitor='val_loss',
                #     factor=0.05,
                #     patience=10,
                #     min_lr=0.00001
                # ),
                tensorboard_callback
            ],
            verbose=1
        )
        
        # Only make predictions if test data was provided
        if use_test_prediction:
            print("Predicting on test data...")
            pred_y = model.predict(test_X, verbose=0)
            pred_y_classes = np.argmax(pred_y, axis=2)
            
            # Restore NaN class labels
            pred_final = np.copy(pred_y_classes)
            pred_final[test_mask == 0] = self.activity_map[np.nan]
            
            # Calculate per-class accuracy
            for class_name, class_id in self.activity_map.items():
                if class_id == self.activity_map[np.nan]:
                    continue
                    
                class_mask = (test_Y_clean == class_id) & (test_mask > 0)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(pred_y_classes[class_mask] == class_id)
                    print(f"{class_name} accuracy: {class_acc:.4f} (n={np.sum(class_mask)})")
        else:
            # No predictions are made (global model)
            pred_final = np.array([])
        
        return model, history, pred_final, test_Y_clean, test_mask

    def _plot_single_history(self, history):
        """Plot training history for a single model"""
        plt.figure(figsize=(12, 4))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save the figure
        output_dir = Path(self.config.analysis.output_dir)
        plt.savefig(output_dir / "lstm_global_history.png", dpi=300)
        plt.close()
        
        # Print final metrics
        print("\nTraining Summary:")
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        if 'val_loss' in history.history:
            print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
            print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")


    def _summarize_results(self, all_predictions, all_actual, all_histories):
        """Summarize and save all model results"""
        # Combine all predictions and actual values
        flat_pred = np.concatenate([p.flatten() for p in all_predictions])
        flat_actual = np.concatenate([a.flatten() for a in all_actual])
        
        # Only evaluate on valid data points (not NaN)
        valid_mask = flat_actual != self.activity_map[np.nan]
        valid_pred = flat_pred[valid_mask]
        valid_actual = flat_actual[valid_mask]
        
        # Calculate overall accuracy
        accuracy = np.mean(valid_pred == valid_actual)
        print(f"\n===== OVERALL RESULTS =====")
        print(f"Total accuracy: {accuracy:.4f}")
        
        # Display class-wise accuracy
        for class_name, class_id in self.activity_map.items():
            if class_id == self.activity_map[np.nan]:
                continue  # Skip NaN class
                
            class_mask = valid_actual == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(valid_pred[class_mask] == class_id)
                print(f"{class_name} accuracy: {class_acc:.4f} (n={np.sum(class_mask)})")
        
        # Save confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(
                valid_actual, 
                valid_pred,
                labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                xticklabels=[self.inv_activity_map[i] for i in sorted([v for k,v in self.activity_map.items() if k is not np.nan])],
                yticklabels=[self.inv_activity_map[i] for i in sorted([v for k,v in self.activity_map.items() if k is not np.nan])]
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save the figure
            output_dir = Path(self.config.analysis.output_dir)
            plt.savefig(output_dir / "lstm_confusion_matrix.png", dpi=300)
            plt.close()
            
            # Plot training history
            self._plot_training_histories(all_histories)
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
        # Save detailed metrics to file
        output_dir = Path(self.config.analysis.output_dir)
        with open(output_dir / "lstm_results.txt", "w") as f:
            f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
            f.write("Class-wise accuracy:\n")
            for class_name, class_id in self.activity_map.items():
                if class_id == self.activity_map[np.nan]:
                    continue
                    
                class_mask = valid_actual == class_id
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(valid_pred[class_mask] == class_id)
                    f.write(f"{class_name}: {class_acc:.4f} (n={np.sum(class_mask)})\n")


    def _plot_training_histories(self, histories):
        """Plot training histories across all folds"""
        plt.figure(figsize=(12, 4))
        
        # Find max length of histories
        max_epochs = max(len(h.history['loss']) for h in histories)
        
        # Initialize arrays for mean and std calculations
        losses = np.zeros((len(histories), max_epochs))
        accuracies = np.zeros((len(histories), max_epochs))
        
        # Fill arrays with padding
        for i, h in enumerate(histories):
            # Pad or truncate loss
            curr_len = len(h.history['loss'])
            losses[i, :curr_len] = h.history['loss']
            losses[i, curr_len:] = h.history['loss'][-1]  # Pad with last value
            
            # Pad or truncate accuracy
            accuracies[i, :curr_len] = h.history['accuracy']
            accuracies[i, curr_len:] = h.history['accuracy'][-1]  # Pad with last value
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        epochs = range(1, max_epochs + 1)
        
        # Plot individual histories
        for i in range(len(histories)):
            plt.plot(epochs[:len(histories[i].history['loss'])], 
                    histories[i].history['loss'], 
                    'b-', alpha=0.1)
        
        # Plot mean and std
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        plt.plot(epochs, mean_loss, 'r-', label='Mean Loss')
        plt.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss, 
                        color='r', alpha=0.2, label='±1 std')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        
        # Plot individual histories
        for i in range(len(histories)):
            plt.plot(epochs[:len(histories[i].history['accuracy'])], 
                    histories[i].history['accuracy'], 
                    'b-', alpha=0.1)
        
        # Plot mean and std
        mean_acc = np.mean(accuracies, axis=0)
        std_acc = np.std(accuracies, axis=0)
        plt.plot(epochs, mean_acc, 'r-', label='Mean Accuracy')
        plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, 
                        color='r', alpha=0.2, label='±1 std')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save the figure
        output_dir = Path(self.config.analysis.output_dir)
        plt.savefig(output_dir / "lstm_training_history.png", dpi=300)
        plt.close()


    def test_isolated_nan_interpolation(self):
        # import pandas as pd
        # import numpy as np
        
        # Create test cases
        test_cases = [
            # Case 1: Isolated NaN in the middle
            [1.0, 2.0, np.nan, 4.0],
            
            # Case 2: Two consecutive NaNs
            [1.0, np.nan, np.nan, 4.0],
            
            # Case 3: NaN at the beginning
            [np.nan, 2.0, 3.0, 4.0],
            
            # Case 4: NaN at the end
            [1.0, 2.0, 3.0, np.nan],
            
            # Case 5: Mixed pattern
            [1.0, np.nan, 3.0, np.nan, np.nan, 6.0],
            
            # Case 6: No NaNs
            [1.0, 2.0, 3.0, 4.0]
        ]
        
        print("Test cases for isolated NaN interpolation:")
        for i, test_data in enumerate(test_cases):
            # Create DataFrame with a single column
            df = pd.DataFrame({'col': test_data})
            print(f"\nTest case {i+1}:")
            print(f"Original: {df['col'].tolist()}")
            
            # Apply standard interpolation
            df_standard = df.copy()
            df_standard['col'] = df_standard['col'].interpolate(method='linear')
            print(f"Standard interpolation: {df_standard['col'].tolist()}")
            
            # Apply our custom approach
            df_custom = df.copy()
            series = df_custom['col']
            mask = series.isna() & series.shift(1).notna() & series.shift(-1).notna()
            if mask.any():
                temp = series.copy()
                temp[~mask] = temp[~mask].fillna(method='ffill')
                temp = temp.interpolate(method='linear')
                series[mask] = temp[mask]
                df_custom['col'] = series
            print(f"Custom interpolation: {df_custom['col'].tolist()}")
        
        return "Tests completed"

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
