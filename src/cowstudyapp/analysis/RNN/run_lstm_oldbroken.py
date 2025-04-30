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


class LSTM_Model:

    def __init__(self, config: ConfigManager):
        self.config = config
        # if config.analysis is not None:
        #     if config.analysis.lstm is not None:
        #         self.config.analysis = config.analysis
        #     else:
        #         raise ValueError("Missing config.analysis.lstm section")
        # else:
        #     raise ValueError("Missing config.analysis section")

        tf.random.set_seed(self.config.analysis.random_seed)
        np.random.seed(self.config.analysis.random_seed)

        # self.df['activity'] = self.df['activity'].map(self.config.analysis.activity_map)

        # self.df = self._get_target_dataset()
        # self._add_step_and_angle()

        # self.normalize_features()

        # self.build_sequences_ops()

        # self.k_fold_validation()
        # self.loocv_ops()

        # self.lmocv_ops_multi()

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
        # print("\nRows before dropping NA:", len(self.df))
        
        # # Drop rows with NA values
        # self.df = self.df.dropna(subset=self.config.analysis.lstm.features)
        
        # print("Rows after dropping NA:", len(self.df))
        
        scaler = StandardScaler()
        
        # print("\nFeature Statistics Before Normalization:")
        # print(self.df[self.config.analysis.features].describe())
        
        # Normalize features
        normalized_data = scaler.fit_transform(df[features])
        df[features] = normalized_data
        
        # print("\nFeature Statistics After Normalization:")
        # print(self.df[self.config.analysis.features].describe())
        
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


        self._build_sequences_opo(df)

    def _build_sequences_opo(self, df):

        max_length = self.config.analysis.lstm.max_length
        sequences:Dict[int, Dict[datetime.date, List[np.ndarray]]] = {}
        
        if max_length == 'daily':
            for device_id, device_data in df.groupby("device_id"):
                sequences[device_id] = {}
                for date, data in device_data.groupby("date"):
                    sequences[device_id][date] = [data[self.config.analysis.lstm.features].to_numpy(), data[['activity']].to_numpy()]



        else:
            pass

        return sequences

    def build_sequences_ops(self, max_length=None):
        """
        LOOK INTO RAGGED
        Build sequences for many-to-one classification with zero padding.
        Each sequence predicts the activity at its last timestamp.
        """

        if not max_length:
            max_length = self.config.analysis.lstm.max_length
        self.sequences = {}
        
        for ID, data in self.df.groupby("ID"):
            sequences = []  # List of observation sequences
            labels = []    # List of activity labels
            current_sequence = []
            
            for i in range(len(data)):
                row = data.iloc[i]
                # Get feature values for current observation
                current_features = [row[f] for f in self.config.analysis.lstm.features]
                
                # Start new sequence if:
                # 1. First record
                # 2. Time gap too large (using posix_time)
                if (i == 0) or (data.iloc[i]['posix_time'] - data.iloc[i-1]['posix_time'] > self.config.analysis.lstm.max_time_gap):
                    current_sequence = []
                
                # Add current observation to sequence
                current_sequence.append(current_features)
                
                # Keep only last max_length observations
                if len(current_sequence) > max_length:
                    current_sequence = current_sequence[-max_length:]
                
                # Create padded sequence
                padded_sequence = np.zeros((max_length, len(self.config.analysis.lstm.features)))
                start_idx = max_length - len(current_sequence)
                padded_sequence[start_idx:] = current_sequence
                
                sequences.append(padded_sequence)
                labels.append(row['activity'])
            
            self.sequences[ID] = {
                'X': np.array(sequences),
                'y': np.array(labels)
            }
            
            # # Print summary statistics
            # print(f"\n-------------------- Collar ID {ID} --------------------")
            # print(f"Total Records: {len(data)}")
            # print(f"Sequences created: {len(sequences)}")
            
            # # Print random sample
            # if len(sequences) > 0:
            #     idx = np.random.randint(0, len(sequences))
            #     print(f"\nRandom sample sequence {idx+1} (shape {sequences[idx].shape}):")
            #     print(f"{sequences[idx]}")
            #     print("Non-zero elements in sequence:", np.count_nonzero(sequences[idx]))
            #     print("Label:", labels[idx])

        return self.sequences


    def loocv_ops(self):
        """Leave-one-out cross validation at the cow level"""
        all_predicted_labels = []
        all_actual_labels = []
        all_histories = []

        n_cows = len(self.sequences.keys())
        print(f"\nStarting Leave-One-Out Cross Validation with {n_cows} cows")
        print("=" * 50)

        for i, test_cow in enumerate(self.sequences.keys()):
            print(f"\nFold {i}/{n_cows} - Testing on cow {test_cow}")
            print("-" * 50)
            test_data = self.sequences[test_cow]['X']
            test_labels = self.sequences[test_cow]['y']

            # Combine all other cows' data for training
            train_data = []
            train_labels = []
            for train_cow in self.sequences.keys():
                if train_cow != test_cow:
                    train_data.append(self.sequences[train_cow]['X'])
                    train_labels.append(self.sequences[train_cow]['y'])
            
            train_data = np.concatenate(train_data)
            train_labels = np.concatenate(train_labels)

            # Train and predict
            model, pred_labels, history = self.build_many_to_one_model(
                train_data, train_labels, test_data, test_labels
            )
            
            all_predicted_labels.extend(pred_labels)
            all_actual_labels.extend(test_labels)
            all_histories.append(history)


        # print("\nActual label distribution:")
        # actual_series = pd.Series(all_actual_labels).map(self.config.analysis.inv_activity_map)
        # print(actual_series.value_counts())


        # print("\nPredicted label distribution:")
        # pred_series = pd.Series(all_predicted_labels).map(self.config.analysis.inv_activity_map)
        # print(pred_series.value_counts())

        # Classification report with string labels
        print("\nOverall Classification Report:")
        print(classification_report(
            all_actual_labels, 
            all_predicted_labels,
            labels=range(len(self.config.analysis.lstm.activity_labels)),
            target_names=self.config.analysis.lstm.activity_labels,
            zero_division=0,
            digits=3
        ))


        # Plot confusion matrix
        self.plot_confusion_matrix(all_actual_labels, all_predicted_labels)
        
        # Plot training histories
        self.plot_training_histories(all_histories)
        
        return all_predicted_labels, all_actual_labels, all_histories


    def k_fold_validation(self, k=5, test_size=0.2):
        """K-fold cross validation with flexible test size"""
        
        # Combine all sequences first
        all_sequences = []
        all_labels = []
        
        for cow_id in self.sequences:
            all_sequences.append(self.sequences[cow_id]['X'])
            all_labels.append(self.sequences[cow_id]['y'])
        
        X = np.concatenate(all_sequences)
        y = np.concatenate(all_labels)
        
        # Create folds
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Track results
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}/{k}")
            print("-" * 30)
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"Training samples: {len(X_train)}")
            print(f"Testing samples: {len(X_test)}")
            
            # Train model
            model, pred_labels, history = self.build_many_to_one_model(
                X_train, y_train, X_test, y_test
            )
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, pred_labels)
            report = classification_report(
                y_test, 
                pred_labels,
                labels=range(len(self.config.analysis.lstm.activity_labels)),
                target_names=self.config.analysis.lstm.activity_labels,
                zero_division=0
            )
            
            print(f"\nFold {fold + 1} Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)
            
            fold_results.append({
                'accuracy': accuracy,
                'predictions': pred_labels,
                'true_labels': y_test,
                'history': history
            })
        
        return fold_results










    def lmocv_ops_multi(self, n_out = 3):
        """Leave-one-out cross validation at the cow level"""
        all_predicted_labels = []
        all_actual_labels = []
        all_histories = []

        n_cows = len(self.sequences.keys())
        print(f"\nStarting Leave-Multiple-Out Cross Validation with {n_cows} cows and ~{n_out} test cows per fold.")
        print("=" * 50)

        n_folds = math.ceil(n_cows/n_out)

        fold_IDs = [[] for _ in range(n_folds)]

        ID_list = list(self.sequences.keys())
        
        np.random.shuffle(ID_list)
        for test_cow in ID_list:
            viable_indices = [i for i in range(len(fold_IDs)) if len(fold_IDs[i]) == min(len(f) for f in fold_IDs)]
            fold_IDs[np.random.choice(viable_indices)].append(test_cow)


        # print(fold_IDs)
        # return

        for i, fold in enumerate(fold_IDs):
            print(f"\nFold {i}/{n_folds} - Testing on cow(s) {fold}")
            test_data = []
            test_labels = []
            
            for c in fold:
                # print(self.sequences[c])
                test_data.append(self.sequences[c]['X'])
                test_labels.append(self.sequences[c]['y'])
            
            test_data = np.concatenate(test_data)
            test_labels = np.concatenate(test_labels)
            
            ###########################################################################

            train_data = []
            train_labels = []
            
            for train_cow in self.sequences.keys():
                if train_cow not in fold:
                    train_data.append(self.sequences[train_cow]['X'])
                    train_labels.append(self.sequences[train_cow]['y'])

            train_data = np.concatenate(train_data)
            train_labels = np.concatenate(train_labels)

            # Train and predict
            model, pred_labels, history = self.build_many_to_one_model(
                train_data, train_labels, test_data, test_labels
            )
            
            all_predicted_labels.extend(pred_labels)
            all_actual_labels.extend(test_labels)
            all_histories.append(history)

        # Classification report with string labels
        print("\nOverall Classification Report:")
        print(classification_report(
            all_actual_labels, 
            all_predicted_labels,
            labels=range(len(self.config.analysis.lstm.activity_labels)),
            target_names=self.config.analysis.lstm.activity_labels,
            zero_division=0,
            digits=3
        ))


        # Plot confusion matrix
        self.plot_confusion_matrix(all_actual_labels, all_predicted_labels)
        
        # Plot training histories
        self.plot_training_histories(all_histories)
        
        return all_predicted_labels, all_actual_labels, all_histories


    def build_many_to_one_model(self, train_x, train_y, test_x, test_y):
        """Build RNN model that predicts single activity from sequence"""
        sequence_length = self.config.analysis.lstm.max_length
        n_features = len(self.config.analysis.lstm.features)
        n_classes = len(self.config.analysis.lstm.states)


        print("\nClass distribution in training data:")
        train_series = pd.Series(train_y).map(self.config.analysis.inv_activity_map)
        print(train_series.value_counts())

        print(np.array(range(n_classes)))
        # Calculate class weights using numeric labels
        n_classes = len(self.config.analysis.activity_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(range(n_classes)),
            y=train_y
        )

        # # Calculate class weights using integer labels directly
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.arange(n_classes),
        #     y=train_y
        # )
        class_weight_dict = dict(zip(range(n_classes), class_weights))


        for i, weight in class_weight_dict.items():
            print(f"{self.config.analysis.activity_labels[i]}: {weight:.3f}")


        model = Sequential([
            Masking(mask_value=0., input_shape=(sequence_length, n_features)),
            
            LSTM(256, 
                return_sequences=True,
                activation='tanh',              # Default activation
                recurrent_activation='sigmoid', # Gate activation
                use_bias=True,
                unit_forget_bias=True,          # Initialize forget gate bias to 1
                recurrent_dropout=0.2,          # Dropout for recurrent connections
                dropout=0.3,                    # Dropout for inputs
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
                recurrent_regularizer=tf.keras.regularizers.L2(0.01)
                ),          
            
            LSTM(128, 
                return_sequences=False,
                recurrent_dropout=0.2,
                dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
                recurrent_regularizer=tf.keras.regularizers.L2(0.01)
                ),          # Dropout for inputs
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])

        # model = Sequential([
        #     Masking(mask_value=0., input_shape=(sequence_length, n_features)),
        #     SimpleRNN(256, return_sequences=True),
        #     Dropout(0.3),
            
        #     SimpleRNN(128, return_sequences=False), #, activation='relu'
        #     Dropout(0.3),
            
        #     Dense(128, activation='relu'),
        #     Dropout(0.3),
        #     # Dense(16, activation='relu'),
        #     Dense(n_classes, activation='softmax')
        # ])




        # learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )


        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,  # Clip gradients to prevent explosions
            clipvalue=0.5
        )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)


        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_x, 
            train_y,
            validation_split=0.2,
            epochs=self.config.analysis.epochs,  # Increase epochs
            batch_size=16,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,          # Let it train longer
                    restore_best_weights=True,
                    min_delta=0.0005     # More sensitive to improvements
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,          # More aggressive reduction
                    patience=7,
                    min_lr=0.00001
                )
            ]
        )        
        # Predict and evaluate
        pred_y = model.predict(test_x, verbose=0)  # Suppress prediction progress bar
        pred_y_classes = np.argmax(pred_y, axis=1)
        # pred_y_labels = le.inverse_transform(pred_y_classes)
        
        # Print metrics in a cleaner format
        # print("\nTest Results:")
        # print("-" * 50)
        
        # # Calculate metrics
        # accuracy = accuracy_score(test_y, pred_y_classes)
        # for i in range(len(pred_y_classes)):
        #     print(f"{i:>3} -- {test_y[i]} -- {pred_y_classes[i]}")
        # cm = confusion_matrix(test_y, pred_y_classes, labels=range(len(self.config.analysis.activity_labels))) #, labels=self.config.analysis.activity_labels
        
        # # Print results
        # print(f"Accuracy: {accuracy:.3f}")
        # print("\nConfusion Matrix:")
        # print("-" * 50)
        # print("True\\Pred", end="\t")
        # print("\t".join(self.config.analysis.activity_labels))
        # for i, row in enumerate(cm):
        #     print(f"{self.config.analysis.activity_labels[i]}", end="\t")
        #     print("\t".join(map(str, row)))
        
        # print("\nDetailed Metrics:")
        # print("-" * 50)
        # report = classification_report(
        #     test_y, 
        #     pred_y_classes,
        #     labels=range(len(self.config.analysis.activity_labels)),  # Use numeric labels
        #     target_names=self.config.analysis.activity_labels,        # Use string names for display
        #     zero_division=0,
        #     digits=3
        # )
        # print(report)
        # # After predictions
        # print("\nPrediction distribution:")
        # print("Unique predicted labels:", np.unique(pred_y_classes, return_counts=True))
        

        ## Print model summary with shapes
        # print("\nModel Architecture:")
        # model.summary()


        return model, pred_y_classes, history




    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix of predictions using numeric labels"""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.config.analysis.activity_labels)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=self.config.analysis.activity_labels,  # Use string labels for display
            yticklabels=self.config.analysis.activity_labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_training_histories(self, histories):
        """Plot training histories across all folds with different lengths"""
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
        
        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nTraining Summary:")
        print(f"Final Mean Loss: {mean_loss[-1]:.4f} ± {std_loss[-1]:.4f}")
        print(f"Final Mean Accuracy: {mean_acc[-1]:.4f} ± {std_acc[-1]:.4f}")
