from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from cowstudyapp.config import ConfigManager, LSTMConfig, AnalysisConfig
from cowstudyapp.utils import from_posix


class LSTM_Model:

    def __init__(self, config: AnalysisConfig):

        self.config = config

        tf.random.set_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # self.df['activity'] = self.df['activity'].map(self.config.activity_map)

        self.get_target_dataset()
        print("target dataset got")
        self.add_step_and_angle()

        self.normalize_features()

        self.build_sequences_ops()

        # self.k_fold_validation()
        # self.loocv_ops()

        self.lmocv_ops_multi()


    def get_target_dataset(self):
        self.df = pd.read_csv(self.config.target_dataset)

    def add_step_and_angle(self):
        """Add step size and turning angle columns to the dataframe"""
        self.df["step"] = None
        self.df["angle"] = None
        
        # Process each cow separately
        for cow_id, cow_data in self.df.groupby("device_id"):
            # Get indices for this cow's data
            cow_indices = cow_data.index
            
            # Calculate step sizes and turning angles for all points except the last one
            for i in range(len(cow_indices)-1):
                curr_idx = cow_indices[i]
                next_idx = cow_indices[i+1]

                # xy_corr,yz_corr,xz_corr,
                # x_peak_to_peak,x_crest_factor,x_impulse_factor,
                # y_peak_to_peak,y_crest_factor,y_impulse_factor,
                # z_peak_to_peak,z_crest_factor,z_impulse_factor,
                # magnitude_peak_to_peak,magnitude_crest_factor,magnitude_impulse_factor,
                # x_mean,x_var,y_mean,y_var,z_mean,z_var,magnitude_mean,magnitude_var,
                # x_entropy,y_entropy,z_entropy,magnitude_entropy,
                # x_dominant_freq,x_dominant_period_minutes, x_spectral_centroid,
                # y_dominant_freq,y_dominant_period_minutes,y_spectral_centroid,
                # z_dominant_freq,z_dominant_period_minutes,z_spectral_centroid,
                # magnitude_dominant_freq,magnitude_dominant_period_minutes,magnitude_spectral_centroid,
                # x_zcr,y_zcr,z_zcr,magnitude_zcr,
                # device_id,posix_time,latitude,longitude,altitude,temperature_gps,dop,satellites,
                # utm_easting,utm_northing,activity

                
                # Current and next points
                x_c, y_c = self.df.loc[curr_idx, "utm_easting"], self.df.loc[curr_idx, "utm_northing"]
                x_n, y_n = self.df.loc[next_idx, "utm_easting"], self.df.loc[next_idx, "utm_northing"]
                
                # Calculate step size
                step = np.sqrt((x_n - x_c)**2 + (y_n - y_c)**2)
                self.df.loc[curr_idx, "step"] = step
                
                # Calculate turning angle (needs three points)
                if i > 0:
                    prev_idx = cow_indices[i-1]
                    # Previous point
                    x_p, y_p = self.df.loc[prev_idx, "utm_easting"], self.df.loc[prev_idx, "utm_northing"]
                    
                    # Calculate vectors
                    vector1 = np.array([x_c - x_p, y_c - y_p])
                    vector2 = np.array([x_n - x_c, y_n - y_c])
                    
                    # Calculate angle between vectors
                    dot_product = np.dot(vector1, vector2)
                    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                    
                    # Avoid division by zero and floating point errors
                    cos_angle = np.clip(dot_product / norms if norms != 0 else 0, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Determine sign of angle using cross product
                    cross_product = np.cross([x_c - x_p, y_c - y_p], [x_n - x_c, y_n - y_c])
                    angle = angle if cross_product >= 0 else -angle
                    
                    self.df.loc[curr_idx, "angle"] = angle
            
            # Set the last row's step and angle to None for this cow
            self.df.loc[cow_indices[-1], ["step", "angle"]] = None

        # Add some validation prints
        print("\nStep and Angle Calculation Summary:")
        print(f"Total rows: {len(self.df)}")
        print(f"Rows with step values: {self.df['step'].notna().sum()}")
        print(f"Rows with angle values: {self.df['angle'].notna().sum()}")
        
        # Show some statistics by cow
        print("\nStatistics by cow:")
        for cow_id, cow_data in self.df.groupby("ID"):
            print(f"\nCow {cow_id}:")
            print(f"Total observations: {len(cow_data)}")
            print(f"Valid steps: {cow_data['step'].notna().sum()}")
            print(f"Valid angles: {cow_data['angle'].notna().sum()}")

    def normalize_features(self):
        """Normalize features before sequence building"""
        print("\nRows before dropping NA:", len(self.df))
        
        # Drop rows with NA values
        self.df = self.df.dropna(subset=self.config.features)
        
        print("Rows after dropping NA:", len(self.df))
        
        scaler = StandardScaler()
        
        # print("\nFeature Statistics Before Normalization:")
        # print(self.df[self.config.features].describe())
        
        # Normalize features
        normalized_data = scaler.fit_transform(self.df[self.config.features])
        self.df[self.config.features] = normalized_data
        
        # print("\nFeature Statistics After Normalization:")
        # print(self.df[self.config.features].describe())
        
        # Store scaler
        self.scaler = scaler

    def build_sequences_ops(self, max_length=None):
        """
        LOOK INTO RAGGED
        Build sequences for many-to-one classification with zero padding.
        Each sequence predicts the activity at its last timestamp.
        """

        if not max_length:
            max_length = self.config.max_length
        self.sequences = {}
        
        for ID, data in self.df.groupby("ID"):
            sequences = []  # List of observation sequences
            labels = []    # List of activity labels
            current_sequence = []
            
            for i in range(len(data)):
                row = data.iloc[i]
                # Get feature values for current observation
                current_features = [row[f] for f in self.config.features]
                
                # Start new sequence if:
                # 1. First record
                # 2. Time gap too large (using posix_time)
                if (i == 0) or (data.iloc[i]['posix_time'] - data.iloc[i-1]['posix_time'] > self.config.max_time_gap):
                    current_sequence = []
                
                # Add current observation to sequence
                current_sequence.append(current_features)
                
                # Keep only last max_length observations
                if len(current_sequence) > max_length:
                    current_sequence = current_sequence[-max_length:]
                
                # Create padded sequence
                padded_sequence = np.zeros((max_length, len(self.config.features)))
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
        # actual_series = pd.Series(all_actual_labels).map(self.config.inv_activity_map)
        # print(actual_series.value_counts())


        # print("\nPredicted label distribution:")
        # pred_series = pd.Series(all_predicted_labels).map(self.config.inv_activity_map)
        # print(pred_series.value_counts())

        # Classification report with string labels
        print("\nOverall Classification Report:")
        print(classification_report(
            all_actual_labels, 
            all_predicted_labels,
            labels=range(len(self.config.activity_labels)),
            target_names=self.config.activity_labels,
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
                labels=range(len(self.config.activity_labels)),
                target_names=self.config.activity_labels,
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
            labels=range(len(self.config.activity_labels)),
            target_names=self.config.activity_labels,
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
        sequence_length = self.config.max_length
        n_features = len(self.config.features)
        n_classes = len(self.config.activity_map.keys())


        print("\nClass distribution in training data:")
        train_series = pd.Series(train_y).map(self.config.inv_activity_map)
        print(train_series.value_counts())

        print(np.array(range(n_classes)))
        # Calculate class weights using numeric labels
        n_classes = len(self.config.activity_labels)
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
            print(f"{self.config.activity_labels[i]}: {weight:.3f}")


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
            epochs=self.config.epochs,  # Increase epochs
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
        # cm = confusion_matrix(test_y, pred_y_classes, labels=range(len(self.config.activity_labels))) #, labels=self.config.activity_labels
        
        # # Print results
        # print(f"Accuracy: {accuracy:.3f}")
        # print("\nConfusion Matrix:")
        # print("-" * 50)
        # print("True\\Pred", end="\t")
        # print("\t".join(self.config.activity_labels))
        # for i, row in enumerate(cm):
        #     print(f"{self.config.activity_labels[i]}", end="\t")
        #     print("\t".join(map(str, row)))
        
        # print("\nDetailed Metrics:")
        # print("-" * 50)
        # report = classification_report(
        #     test_y, 
        #     pred_y_classes,
        #     labels=range(len(self.config.activity_labels)),  # Use numeric labels
        #     target_names=self.config.activity_labels,        # Use string names for display
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
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.config.activity_labels)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=self.config.activity_labels,  # Use string labels for display
            yticklabels=self.config.activity_labels
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