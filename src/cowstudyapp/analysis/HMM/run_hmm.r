# run_hmm.R

init_the_script <- function(util_path, config_path, target_dataset_path, cv_dir, mod_dir, pred_dir){


    
    # Load dependencies
    cat("Loading dependencies...\n")
    
    packages <- c("momentuHMM", "dplyr", "ggplot2", "caret", "fitdistrplus",
                 "circular", "CircStats", "lubridate", 'grid', 'gridExtra',
                 "movMF", "suncalc", "fs")

    suppressWarnings({
        suppressPackageStartupMessages({
            invisible(lapply(packages, library, character.only = TRUE))
        })
    })

    # 
    cat("Sourcing utility functions from:", util_path, "\n")
    source(util_path)
    
    # Load configuration
    cat("Loading configuration from:", config_path, "\n")
    
    config <- jsonlite::fromJSON(config_path)
    Options <- config$options
    features <- config$features
    states <- config$states
    
    set.seed(config$random_seed)

    all_activities <- config$all_activities
    # all_activities <- c("Grazing", "Resting", "Traveling", "Drinking", "Fighting", "Mineral", "Scratching")


    # dirs <- create_output_structure(base_output_dir=output_dir, features=features)

    cv_dir <- path(cv_dir)
    mod_dir <- path(mod_dir)
    pred_dir <- path(pred_dir)

    #     plots_dir = file.path(output_dir, "plots"),
    #     dist_plots_dir = file.path(output_dir, "plots", "distributions")
    # )
    
    cat("Loading data from:", target_dataset_path, "\n")
    cat("   States:", paste(states, collapse=", "), "\n")
    cat("   Features:", features$name, "\n")

    if (!is.null(Options)) {
        cat("\nOptions:\n")
        for (name in names(Options)) {
            if (is.list(Options[[name]])) {
                # If it's a list, print each element separately
                cat("- ", name, ":\n", sep="")
                for (subname in names(Options[[name]])) {
                    # Handle nested lists (like distributions)
                    if (is.list(Options[[name]][[subname]]) || is.vector(Options[[name]][[subname]]) && length(Options[[name]][[subname]]) > 1) {
                        cat("    - ", subname, ": [", sep="")
                        cat(paste(Options[[name]][[subname]], collapse=", "))
                        cat("]\n")
                    } else {
                        cat("  - ", subname, ": ", Options[[name]][[subname]], "\n", sep="")
                    }
                }
            } else {
                # If it's not a list, print normally
                cat("- ", name, ": ", Options[[name]], "\n", sep="")
            }
        }
    }

    cat("Selecting and renaming target data...\n")
    target_dataset <- load_csv_data(target_dataset_path, features)

    cat("Ensuring cows in Excluded devices are not in the dataset...\n")
    target_dataset <- subset(target_dataset, !(ID %in% config$excluded_devices))


    # cat(config$mode)
    if (config$mode == "LOOCV") {
        # Run LOOCV on target dataset
        cat("Starting LOOCV...\n")
        run_LOOCV(config, Options, features, states, cv_dir, target_dataset, all_activities)
        

    }else if(config$mode == "PRODUCT"){
        # str(config$training_info)
        if (!is.null(config$training_info)) {

            cat(sprintf("PRODUCT -- config$training_info$type = %s\n",config$training_info$training_info_type))
            if (config$training_info$training_info_type == "dataset") {
                # Train new model using training dataset
                cat("Configuration set to use 'dataset' as the training type\n")

                cat("Preparing target data as a momentuHMM object...\n")
                target_dataset <- prepare_hmm_data(data=target_dataset, states=states, config=config)
                
                cat("Loading the training dataset...\n")
                training_dataset <- load_csv_data(data_path=config$training_info$training_info_path
                                                , features=features, all_activities=all_activities)

                training_dataset <- subset(training_dataset, !(ID %in% config$excluded_devices))
                cat("Creating the training prepData object...\n")
                training_dataset <- prepare_hmm_data(data=training_dataset, states=states, config=config)
                # validate_training_data(training_data)
                
                cat("Selecting the labeled data from the training dataset...\n")
                training_dataset <- process_labeled_data(training_dataset, states, all_activities, config$timezone)

                cat("Computing features from the training data...\n")
                features <- select_best_distributions(data=training_dataset, 
                                    features=features,
                                    Options=Options,
                                    dir=mod_dir)


                cat("Training the model...\n")
                trained_model <- train_model(training_dataset, features, states, time_covariate=config$time_covariate)
                
            } else if (config$training_info$training_info_type == "model") {
                cat("Configuration set to use 'model' as the training type\n")

                # cat("Target Data head:\n")
                # print(head(target_dataset))

                cat("Preparing target data as a momentuHMM object...\n")
                target_dataset <- prepare_hmm_data(data=target_dataset, states=states, config=config)
                # training_dataset <- subset(training_dataset, !(ID %in% config$excluded_devices))
   
                # Load and apply existing model
                cat("Loading model parameters...\n")
                params <- load_model_parameters(config$training_info$training_info_path)
                features <- params$features
                trained_model <- params$model 

            } else {
                stop(sprintf("Invalid training_info type `%s`. Must be 'dataset' or 'model'.", config$training_info$type))
            }


            cat("Applying trained model to target dataset...\n")
            model <- apply_model(target_dataset, trained_model, features, states, time_covariate=config$time_covariate)
            
            cat("Getting target dataset predictions...\n")
            # get_predictions_from_model(model, target_dataset, dirs, states)
            get_predictions_from_model(model, pred_dir, config)
        
        # # Make predictions. Read the 
        # viterbi(model)
        
        } else {
            # cat("No training info provided. Using the Target data\n")

            cat("Using the target dataset as the training dataset...\n")

            cat("Preparing target data as a momentuHMM object...\n")
            target_dataset <- prepare_hmm_data(data=target_dataset, states=states, config=config)
        
            cat("Loading the training dataset...\n")
            training_data <- load_csv_data(data_path=target_dataset_path,features=features,all_activities=all_activities)
            
            cat("Creating the training prepData object...\n")
            training_data <- prepare_hmm_data(data=training_data, states=states, config=config)
            
            cat("Selecting the labeled data from the training dataset...\n")
            training_data <- process_labeled_data(raw_data=training_data, states=states, all_activities=all_activities, timezone=config$timezone)

            cat("Computing features from the training data...\n")
            features <- select_best_distributions(data=training_data, 
                                features=features,
                                Options=Options,
                                dir=mod_dir)

            cat("Training the model...\n")
            trained_model <- train_model(training_data, features, states, time_covariate=config$time_covariate)

            cat("Applying trained model to target dataset...\n")
            model <- apply_model(target_dataset, trained_model, features, states, time_covariate=config$time_covariate)
            
            cat("Getting target dataset predictions...\n")
            # get_predictions_from_model(model, target_dataset, dirs, states)
            get_predictions_from_model(model, pred_dir, config)

            cat("Saving the model and featureset to disk...\n")
            save_model(model, mod_dir, features)

        }
        cat("Finished!...\n")
    
    }else{
        stop(sprintf("Invalid mode `%s` selected.\n", config.mode))
    }
}


load_csv_data <- function(data_path, features, all_activities){
    # Load, rename, and select data
    df <- read.csv(data_path)

    # Column mapping
    # print(1)
    # Start with the base required mappings
    column_mapping <- list(
        "device_id" = "ID",
        "posix_time" = "posix_time",
        "utm_easting" = "x",
        "utm_northing" = "y",
        "longitude" = "longitude",
        "latitude" = "latitude",
        "activity" = "activity"
    )

    for (i in 1:nrow(features)) {
        feature_name <- features$name[i]
        column_mapping[[feature_name]] <- feature_name
    }


    # Define features to ignore for validation (prepdata created features)
    ignore_features <- c("step", "angle")

    # Rename columns
    for (old_name in names(column_mapping)) {
        # Skip if the column is in the ignore list
        if (old_name %in% ignore_features) {
            next
        }
        # Attempt to rename if column exists
        if (old_name %in% names(df)) {
            names(df)[names(df) == old_name] <- column_mapping[[old_name]]
        } else {
            warning(sprintf("Column '%s' not found in data", old_name))
        }
    }

    # Get required columns, excluding ignored features
    required_columns <- unlist(column_mapping[!names(column_mapping) %in% ignore_features])
    
    # Check for missing columns
    missing_columns <- required_columns[!required_columns %in% names(df)]
    if (length(missing_columns) > 0) {
        stop(sprintf("Missing required columns: %s", 
                    paste(missing_columns, collapse=", ")))
    }
    # print(4)

    df <- df[, required_columns]
    df$ID <- as.integer(df$ID)

    # str(df)
    return(df)
}

load_model_parameters <- function(model_path) {
    param_file <- file.path(model_path)
    if (!file.exists(param_file)) {
        stop("Model parameters file not found: ", param_file)
    }
    readRDS(param_file)
}

save_model <- function(model, base_dir, features){
    # Save the full model
    model_file <- file.path(base_dir, "trained_model.rds")

    s <- list(features=features, model=model)
    saveRDS(s, model_file)
    cat(sprintf("Saved full model to: %s\n", model_file))
}

get_predictions_from_model <- function(model, output_dir, config) {
    tryCatch({
    timezone <- "America/Denver"  # Let's be explicit about timezone
    predictions <- viterbi(model)
    states <- model$stateNames
    results_df <- model$data
    results_df$predicted_state <- states[predictions]
    results_df$factored_activity <- NULL
    buffer <- 1.5
    
    if(config$day_only){
        # Convert to Mountain time consistently
        utc_times <- as.POSIXct(results_df$posix_time, origin="1970-01-01", tz="UTC")
        mtn_times <- with_tz(utc_times, timezone)
        # Convert to dates explicitly using format and as.Date
        mtn_dates <- as.Date(format(mtn_times, "%Y-%m-%d"))
        unique_dates <- unique(mtn_dates)
        unique_ID <- unique(results_df$ID)
        
        for (id in unique_ID){
        for(date_str in unique_dates) {
            # Convert string to Date object for getSunlightTimes
            date <- as.Date(date_str)
            # Use Mountain time dates for masking
            mask <- (mtn_dates == date & results_df$ID == id)
            filtered_data <- results_df[mask,]
            day_data <- data.frame(
            date = date,  # This should now be a proper Date object
            lat = filtered_data$latitude[1],
            lon = filtered_data$longitude[1]
            )
            sun_times <- getSunlightTimes(
            data = day_data,
            keep = c("sunrise", "sunset"),
            tz = timezone
            )
            # Create night mask using Mountain times
            night_mask <- mask & (
            mtn_times < (sun_times$sunrise - (60*60)*buffer) |
            mtn_times > (sun_times$sunset + (60*60)*buffer)
            )
            # Debug output
            daily_times <- mtn_times[mask]
            results_df$activity[night_mask] <- "NIGHTTIME"
            results_df$predicted_state[night_mask] <- "NIGHTTIME"
        }
        }
    }            
    
    # Filter for rows with actual labels
    labeled_data <- results_df[(!is.na(results_df$activity))&(results_df$activity != "NIGHTTIME") , ]
    
    # Create confusion matrix
    conf_matrix <- table(Actual = labeled_data$activity,
                        Predicted = labeled_data$predicted_state)
    
    # Calculate metrics for each state
    metrics_list <- lapply(states, function(state) {
        # Convert to binary classification for each state
        actual_binary <- labeled_data$activity == state
        pred_binary <- labeled_data$predicted_state == state
        
        # Calculate metrics
        TP <- sum(actual_binary & pred_binary)
        TN <- sum(!actual_binary & !pred_binary)
        FP <- sum(!actual_binary & pred_binary)
        FN <- sum(actual_binary & !pred_binary)
        
        # Compute statistics
        accuracy <- (TP + TN) / (TP + TN + FP + FN)
        sensitivity <- TP / (TP + FN)
        specificity <- TN / (TN + FP)
        precision <- TP / (TP + FP)
        f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
        
        # Handle NaN cases
        metrics <- c(
        accuracy = ifelse(is.nan(accuracy), 0, accuracy),
        sensitivity = ifelse(is.nan(sensitivity), 0, sensitivity),
        specificity = ifelse(is.nan(specificity), 0, specificity),
        f1_score = ifelse(is.nan(f1_score), 0, f1_score)
        )
        return(metrics)
    })
    names(metrics_list) <- states
    
    # Create metrics data frame for table display
    metrics_df <- do.call(rbind, metrics_list)
    metrics_df <- as.data.frame(metrics_df)
    metrics_df$State <- rownames(metrics_df)
    metrics_df <- metrics_df[, c("State", "accuracy", "sensitivity", "specificity", "f1_score")]
    colnames(metrics_df) <- c("State", "Accuracy", "Sensitivity", "Specificity", "F1 Score")
    
    # Calculate overall accuracy for display
    overall_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
    
    # Print results to terminal with nicely formatted tables
    cat("\nPerformance Metrics by State:\n")
    cat("============================\n")
    print(format(metrics_df, digits = 3))
    
    cat("\nOverall Accuracy:", round(overall_accuracy, 3), "\n")
    
    cat("\nConfusion Matrix:\n")
    print(conf_matrix)
    
    # Save metrics to file using the same table formatting
    sink(file.path(output_dir, "performance_metrics.txt"))
    
    cat("Confusion Matrix:\n")
    cat("=================\n")
    print(conf_matrix)
    
    cat("\nPerformance Metrics by State:\n")
    cat("============================\n")
    print(format(metrics_df, digits = 3))
    
    cat("\nOverall Accuracy:", round(overall_accuracy, 3), "\n")
    sink()
    
    }, error = function(e) {
    cat("\nError in viterbi:", e$message)
    cat("\nModel state at error:")
    # str(model)
    stop(e)
    })

    # Save predictions to CSV
    results_df$magnitude_var <- expm1(results_df$magnitude_var)
    predictions_file <- file.path(output_dir, "predictions.csv")
    write.csv(results_df, predictions_file, row.names = FALSE)
    cat("\nPredictions saved to:", predictions_file)
    
    # Generate and save model parameters summary
    sink(file.path(output_dir, "model_parameters.txt"))
    cat("\nModel Parameters Summary")
    cat("\n=====================\n")
    
    # Transition probability matrix
    cat("\nTransition Probability Matrix:\n")
    print(model$mle$gamma)
    
    # Initial state distribution
    cat("\nInitial State Distribution:\n")
    print(model$mle$delta)
    
    # Distribution parameters for each feature
    cat("\nFeature Distribution Parameters:\n")
    for (feature in names(model$mle)) {
        if (!(feature %in% c("gamma", "delta"))) {
            cat(sprintf("\n%s:\n", feature))
            print(model$mle[[feature]])
        }
    }
    
    # Generate summary statistics of predictions
    cat("\nPredicted State Distribution:\n")
    state_counts <- table(predictions)
    state_props <- prop.table(state_counts)
    summary_stats <- data.frame(
        State = states,
        Count = as.vector(state_counts),
        Percentage = round(as.vector(state_props) * 100, 2)
    )
    print(summary_stats)
    
    # Time-based statistics
    cat("\nAverage State Duration (in seconds):\n")
    current_state <- predictions[1]
    current_count <- 1
    durations <- list()
    for (state in states) {
        durations[[state]] <- numeric()
    }
    
    for (i in 2:length(predictions)) {
        if (predictions[i] != current_state || i == length(predictions)) {
            durations[[states[current_state]]] <- c(
                durations[[states[current_state]]], 
                current_count * 300  # Assuming 5-minute intervals (300 seconds)
            )
            current_state <- predictions[i]
            current_count <- 1
        } else {
            current_count <- current_count + 1
        }
    }
    
    duration_stats <- data.frame(
        State = states,
        Mean_Duration_Sec = sapply(durations, mean),
        Median_Duration_Sec = sapply(durations, median),
        Max_Duration_Sec = sapply(durations, max)
    )
    print(duration_stats)
    
    sink()
    
    cat("\nModel parameters and statistics saved to:", 
        file.path(output_dir, "model_parameters.txt"))
    cat("\nPrediction generation complete.\n")
    
    return(results_df)
}



validate_config <- function(config, target_dataset) {
    #Annoying, Kind of redundant
    # Basic path validation
    # if (!file.exists(data_path)) {
    #     stop("Target dataset not found: ", data_path)
    # }
    
    # Mode-specific validation
    if (config$mode == "LOOCV") {
        # Check if data has labels
        # data <- read.csv(data_path)
        if (all(is.na(target_dataset$activity))) {
            stop("LOOCV mode requires labeled data")
        }
    } else if (config$mode == "PRODUCT") {
        if (!is.null(config$training_info)) {
            if (config$training_info$training_info_type == "dataset") {
                if (!file.exists(config$training_info$path)) {
                    stop("Training dataset not found: ", config$training_info$training_info_path)
                }
            } else if (config$training_info$training_info_type == "model") {
                model_path <- file.path(config$training_info$training_info_path)
                if (!file.exists(model_path)) {
                    stop("Model parameters not found: ", model_path)
                }
            }
        } else {
            # No training info - check if target data has labels
            # data <- read.csv(data_path)
            if (all(is.na(target_dataset$activity))) {
                stop("When no training info is provided, target dataset must have labels")
            }
        }
    }
}





process_labeled_data <- function(raw_data, states, all_activities, timezone) {
    # Basic datetime conversion
    # print("22!!!!!!!!!!!!!!!!!!!!")

    # print(head(raw_data))
    # print(states)
    # print(all_activities)
    # print(timezone)


    raw_data$datetime <- as.POSIXct(raw_data$posix_time, 
                                   origin="1970-01-01", 
                                   tz="UTC")
    raw_data$datetime <- with_tz(raw_data$datetime, timezone)
    
    # print("22!!!!!!!!!!!!!!!!!!!!")
    # Filter for valid activities
    filtered_data <- raw_data[raw_data$activity != "", ]
    # print("22!!!!!!!!!!!!!!!!!!!!")
    # # print(length(filtered_data$ID))
    # # print(all_activities)
    # print(unique(raw_data$activity))
    # print(unique(raw_data$ID))

    # # Print activity value counts
    # cat("\nActivity value counts in raw_data:\n")
    # activity_counts <- table(raw_data$activity)
    # print(activity_counts)
    
    # # Print percentage of each activity
    # activity_percentages <- prop.table(activity_counts) * 100
    # cat("\nActivity percentages:\n")
    # for(act in names(activity_counts)) {
    #     cat(sprintf("%s: %d (%.2f%%)\n", 
    #                act, 
    #                activity_counts[act], 
    #                activity_percentages[act]))
    # }

    
    filtered_data <- filtered_data[order(filtered_data$ID, filtered_data$posix_time), ]

    # print("22!!!!!!!!!!!!!!!!!!!!")
    # Convert activities
    filtered_data$activity <- ifelse(filtered_data$activity %in% states,
                                   filtered_data$activity,
                                   NA)
    
    # print("22!!!!!!!!!!!!!!!!!!!!")
    return(filtered_data)
}



create_time_covariate <- function(posix_time, latitude, longitude, timezone) {
    datetime <- as.POSIXct(posix_time, origin="1970-01-01", tz=timezone)
    
    # Get unique dates to calculate sun times once per date
    unique_dates <- unique(as.Date(datetime))
    sun_times_lookup <- data.frame()
    
    # Calculate sun times for each unique date
    for(date in unique_dates) {
        day_data <- data.frame(
            date = as.Date(date),
            lat = latitude[1],  # Assuming location doesn't change much
            lon = longitude[1]
        )
        
        day_sun_times <- getSunlightTimes(data = day_data, 
                                        keep = c("sunrise", "sunset"), 
                                        tz = timezone)
        
        sun_times_lookup <- rbind(sun_times_lookup, day_sun_times)
    }
    
    # Match sun times to each datetime
    dates <- as.Date(datetime)
    sunrise_times <- sun_times_lookup$sunrise[match(dates, sun_times_lookup$date)]
    sunset_times <- sun_times_lookup$sunset[match(dates, sun_times_lookup$date)]
    
    # Convert to hours
    hours <- as.numeric(format(datetime, "%H")) + 
             as.numeric(format(datetime, "%M"))/60
    
    sunrise_hours <- as.numeric(format(sunrise_times, "%H")) +
                    as.numeric(format(sunrise_times, "%M"))/60
    sunset_hours <- as.numeric(format(sunset_times, "%H")) +
                    as.numeric(format(sunset_times, "%M"))/60
    
    morning_peak_time <- sunrise_hours + 1.5
    evening_peak_time <- sunset_hours - 1.5

    time_normalized <- (hours / 24) * 2 * pi
  
    mu1 <- mean((morning_peak_time / 24) * 2 * pi)
    mu2 <- mean((evening_peak_time / 24) * 2 * pi)
    day_center <- (mu1 + mu2) / 2
    
    peak1 <- 1*dvonmises(circular(time_normalized), mu = mu1, kappa = 6)
    peak2 <- 1*dvonmises(circular(time_normalized), mu = mu2, kappa = 6)
    peak3 <- .8*dvonmises(circular(time_normalized), mu = day_center, kappa = 1)

    # Combine peaks
    diurnal_score <- ((peak1 + peak2) / 2) + peak3 
    
    # Normalization
    final_score <- diurnal_score / max(diurnal_score, na.rm = TRUE)
    # final_score <- diurnal_score
    return(final_score)
}

prepare_hmm_data <- function(data, states, time_covariate, config) {
    # Sometimes the extreme outliers of mag_var cause issues with overflow. A kind of crude but
    # effective fix is to substitue those values with unlikely, but relatively more likely values.  

    max_magnitude_var <- 8
    if ("magnitude_var" %in% names(data)) {


        # outliers <- data$magnitude_var > max_magnitude_var
        # if (any(outliers)) {
        #     random_range <- 10
        #     n_outliers <- sum(outliers)
        #     random_values <- max_magnitude_var + rexp(n_outliers, 1/2)
        #     data$magnitude_var[outliers] <- random_values
            
        #     ## Fixed cutoff value
        #     # data$magnitude_var <- pmin(data$magnitude_var, max_magnitude_var)
        # }

        
        # Optional: Add log transform to compress the range
        data$magnitude_var <- log1p(data$magnitude_var)  # log1p(x) = log(1 + x)

    }
    # max_magnitude_var = 8
    # if ("magnitude_var" %in% names(data)) {
    #     data$magnitude_var <- pmin(data$magnitude_var, max_magnitude_var)
        
    #     # Optional: Add log transform to compress the range
    #     data$magnitude_var <- log1p(data$magnitude_var)  # log1p(x) = log(1 + x)

    # }

    # str(data)
    # cat("states!!!!!!!!!!: ",unique(data$activity),"\n")
    # cat("factor_levels!!!!!!!!!!: ",states,"\n")
    prepped_data <- prepData(data)
    prepped_data$factored_activity <- as.integer(factor(data$activity, 
                                                      levels = states))

    # str(prepped_data)


    if (config$time_covariate == TRUE){
        cat("Adding time_covariate to data...\n")
        prepped_data$time_covariate <- create_time_covariate(
            posix_time = prepped_data$posix_time,
            latitude = prepped_data$latitude,
            longitude = prepped_data$longitude,
            timezone = config$timezone
    )}


    return(prepped_data)
} 



train_model <- function(prepped_data, features, states, time_covariate) {
    
    features$has_zeros <- sapply(features$name, function(fname) {
        any(prepped_data[[fname]] == 0, na.rm = TRUE)
    })


    state_ids <- 1:length(states)
    data_by_state <- list()
    for (i in seq_along(states)) {
        data_by_state[[states[i]]] <- subset(prepped_data, factored_activity == state_ids[i])
    }
    
    parameters <- list()

    for (i in 1:nrow(features)) {
    parameters[[features$name[i]]] <- calculate_parameters(
        data_by_state = data_by_state,
        feature_name = features$name[i],
        dist_type = features$dist[i],
        nbStates = length(states),
        feature_has_zeros = features$has_zeros[i]
    )}

    if (time_covariate == TRUE){
        f = ~prepped_data$time_covariate
    }else{
        f = ~1
    }

    model <- fitHMM(data = prepped_data,
                   nbStates = length(states),
                   dist = get_feature_distributions(features),
                   Par0 = parameters,
                   stateNames = states,
                   knownStates = prepped_data$factored_activity,
                   formula = f,
                   estAngleMean = list(angle = TRUE))
    
    return(model)
}



apply_model <- function(data, trained_model, features, states, time_covariate) {
    Par0 <- getPar0(trained_model)
    
    if (!is.null(data$activity)) {
        data$activity <- ifelse(data$activity %in% states,
                              data$activity,
                              NA)
    }    
    # print(features)
    dist_list <- get_feature_distributions(features)
    # print(dist_list)

    zeromass_imputed_features <- features

    # Check for zeros in the new dataset
    zeromass_imputed_features$has_zeros <- sapply(zeromass_imputed_features$name, function(fname) {
        any(data[[fname]] == 0, na.rm = TRUE)
    })
    


    # Modify Par0 parameters based on zero presence
    modified_Par0 <- Par0$Par
    for (fname in zeromass_imputed_features$name) {
        dist_type <- zeromass_imputed_features$dist[which(zeromass_imputed_features$name == fname)]
        if (dist_type %in% c("gamma", "weibull", "exp", "lnorm")) {
            trained_has_zeros <- length(modified_Par0[[fname]]) > 2 * length(states)
            current_has_zeros <- zeromass_imputed_features$has_zeros[which(zeromass_imputed_features$name == fname)]
            
            if (current_has_zeros && !trained_has_zeros) {
                n_params <- length(modified_Par0[[fname]])
                params_per_state <- n_params / length(states)
                zeromass <- rep(0.0001, length(states))
                modified_Par0[[fname]] <- c(modified_Par0[[fname]], zeromass)
            } else if (!current_has_zeros && trained_has_zeros) {
                n_params <- length(modified_Par0[[fname]])
                params_per_state <- (n_params / length(states)) - 1
                modified_Par0[[fname]] <- modified_Par0[[fname]][1:(params_per_state * length(states))]
            }
        }
    }

    if (time_covariate == TRUE){
        f = ~data$time_covariate
    }else{
        f = ~1
    }

    # Create model without fitting, and copy the trained parameters
    model <- fitHMM(data = data,
                   nbStates = length(states),
                   dist = dist_list,
                   Par0 = modified_Par0,
                   beta0 = Par0$beta,
                   delta0 = Par0$delta,
                   stateNames = states,
                   formula = f,
                   estAngleMean = list(angle = TRUE),
                   fit = FALSE)

    # cat("\n3. New model MLE parameters:\n")
    # str(model$mle)
    
    # cat("\n4. Model structure check:")
    # cat("\n   - States:", length(model$stateNames))
    # cat("\n   - Features:", names(model$mle))
    # cat("\n   - Gamma matrix dimensions:", dim(model$mle$gamma))
    # cat("\n   - Delta vector length:", length(model$mle$delta))
    # str(model)
    return(model)
}


run_LOOCV <- function(config, Options, features, states, cv_dir, target_dataset, all_activities) {

    cat("Starting HMM analysis...\n")


    # print("1!!!!!!!!!!!!!!!!!")
    # print(length(target_dataset$ID))

    prepped_data <- prepare_hmm_data(target_dataset, states, config=config)

    prepped_data <- process_labeled_data(prepped_data, states, all_activities, config$timezone)

    cat("Selecting feature distributions...\n")

    features <- select_best_distributions(data=prepped_data, 
                                          features=features,
                                          Options=Options,
                                          dir=cv_dir)

    cat("Features After Selecting best distributions:\n")
    for (i in 1:nrow(features)) {
        cat(sprintf("   %s: %s\n", 
                    features$name[i], 
                    ifelse(is.na(features$dist[i]), "not set", features$dist[i])))
    }

    # Show correlation analysis if requested
    if (Options$show_correlation) {
        correlation_plot <- plot_correlation_matrix(data = prepped_data, 
                                                    features = features)
        # plots_dir <- file.path(dir, "distributions")
        # dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

        save_plot(correlation_plot, 
                 "correlation_matrix", 
                 cv_dir)
    }

    # Do LOOCV
    cat("Starting LOOCV...\n")
    
    state_ids <- 1:length(states)
    all_actual_states <- c()
    all_predicted_states <- c()


    cv_models <- list()
    unique_cows <- unique(prepped_data$ID)
    # unique_cows <- unique(prepped_data$Base_ID)
    cv_models <- vector("list", length(unique_cows))
    names(cv_models) <- as.character(unique_cows)

    # Create a data frame to store per-cow results
    cow_results <- data.frame(
        cow_id = unique_cows,
        n_labels = 0,
        n_predictions = 0,
        accuracy = 0
    )
    
    # Add columns for each state's predictions
    for (state in states) {
        cow_results[paste0("correct_", tolower(state))] <- 0
        cow_results[paste0("total_", tolower(state))] <- 0
    }

    # Initialize cow counter
    cow_counter <- 1
    
    # for (test_cow in head(unique_cows,2)) {
    for (test_cow in unique_cows) {
        cat("+---------------------------------------------------------------------+\n                          LOOCV with test cow", test_cow, "\n")

        test_data <- subset(prepped_data, ID == test_cow)
        train_data <- subset(prepped_data, ID != test_cow)
        
        # Split data by state
        data_by_state <- list()
        for (i in seq_along(states)) {
            data_by_state[[states[i]]] <- subset(train_data, factored_activity == state_ids[i])
        }
        combined_data <- rbind(train_data, test_data)
        
        # Calculate parameters for each feature
        parameters <- list()

        features$has_zeros <- sapply(features$name, function(fname) {
            any(combined_data[[fname]] == 0, na.rm = TRUE)
        })

        cat("Calculating parameters...\n")
        for (i in 1:nrow(features)) {
            if (features$dist_type[i] != "covariate") {
                parameters[[features$name[i]]] <- calculate_parameters(
                    data_by_state = data_by_state,
                    feature_name = features$name[i],
                    dist_type = features$dist[i],
                    nbStates = length(states),
                    feature_has_zeros = features$has_zeros[i]
                )
            }
        }

        if (config$time_covariate == TRUE){
            f = ~combined_data$time_covariate
        }else{
            f = ~1
        }

        cat("Fitting the model...\n")
        model <- fitHMM(data = combined_data,
                       nbStates = length(states),
                       dist = get_feature_distributions(features),
                       Par0 = parameters,
                       stateNames = states,
                       knownStates = c(train_data$factored_activity, rep(NA, nrow(test_data))),
                       formula = f,
                       estAngleMean = list(angle = TRUE))

        cat("Running viterbi...\n")
        predicted_states <- viterbi(model)

        # Extract predictions for the test data
        test_predicted_states <- predicted_states[(nrow(train_data) + 1):nrow(combined_data)]
        
        # Calculate statistics for this cow
        valid_indices <- !is.na(test_data$factored_activity)

        

        test_actual <- test_data$factored_activity[valid_indices]
        test_pred <- test_predicted_states[valid_indices]
        cow_factored_actual <- factor(states[test_actual], levels = states)
        cow_factored_pred <-factor(states[test_pred], levels = states)

        # print(data.frame(list(
        #     actual=states[test_actual],
        #     pred=states[test_pred])
        # ))

        summary_table <- confusionMatrix(cow_factored_pred,cow_factored_actual)
        print(summary_table)
        # str(summary_table)
        cat("\n-----------------------------\n")
        # cat(summary_table$table)
        # cat("\n")

        # Store results for this cow using cow_counter
        cow_results$n_labels[cow_counter] <- sum(valid_indices)
        cow_results$n_predictions[cow_counter] <- length(test_predicted_states)

        if (sum(valid_indices) > 0) {
            cow_results$accuracy[cow_counter] <- mean(test_actual == test_pred, na.rm = TRUE)

            # Calculate correct predictions for each state
            for (state_idx in seq_along(states)) {
                state <- states[state_idx]
                state_actual <- test_actual == state_idx
                state_pred <- test_pred == state_idx
                
                correct_predictions <- sum(state_actual & (test_actual == test_pred), na.rm = TRUE)
                total_actual <- sum(state_actual, na.rm = TRUE)
                
                cow_results[cow_counter, paste0("correct_", tolower(state))] <- correct_predictions
                cow_results[cow_counter, paste0("total_", tolower(state))] <- total_actual
            }
        }
        
        # Store model
        cv_models[[cow_counter]] <- model
        all_actual_states <- c(all_actual_states, test_actual)
        all_predicted_states <- c(all_predicted_states, test_pred)
        
        # Print individual cow results
        cat("\nResults for cow", test_cow, ":\n")
        print(format_cow_results(cow_results[cow_counter,]))
        
        # Increment cow counter
        cow_counter <- cow_counter + 1
    
    }
    cat("Finished LOOCV\n")

 # Print final summary table
    cat("\n=== Complete LOOCV Summary ===\n")
    conf_matrix <- confusionMatrix(all_predicted_states, all_actual_states)
    
    # Calculate and print overall statistics
    cat("\n=== Overall Statistics ===\n")
    
    # Calculate overall accuracy excluding cows with no labels
    valid_cows <- cow_results$n_labels > 0
    overall_accuracy <- mean(cow_results$accuracy[valid_cows], na.rm = TRUE)
    cat(sprintf("Average Accuracy: %.2f%%\n", overall_accuracy * 100))
    
    # Calculate state-specific accuracies
    for (state in states) {
        correct_col <- paste0("correct_", tolower(state))
        total_col <- paste0("total_", tolower(state))
        total_correct <- sum(cow_results[[correct_col]], na.rm = TRUE)
        total_cases <- sum(cow_results[[total_col]], na.rm = TRUE)
        
        state_accuracy <- if(total_cases > 0) {
            (total_correct / total_cases) * 100
        } else {
            NA
        }
        
        if (!is.na(state_accuracy)) {
            cat(sprintf("%s Accuracy: %.2f%%\n", state, state_accuracy))
        } else {
            cat(sprintf("%s Accuracy: No valid cases\n", state))
        }
    }



    # Ensure both actual and predicted states are factors with the same levels
    # print("!!!!!!")
    # print(all_actual_states)

    valid_indices <- !is.na(all_actual_states)

    # Filter both vectors using these indices
    # all_actual_states <- states[all_actual_states[valid_indices]]
    all_actual_states <- factor(states[all_actual_states[valid_indices]], levels = states)
    
    # print("!!!!!!!!!!!!!")
    # print(all_actual_states)


    # all_predicted_states1 <- states[all_predicted_states[valid_indices]]
    all_predicted_states1 <- factor(states[all_predicted_states[valid_indices]], levels = states)
    # print(all_predicted_states1)

    # print("!!!!!!!!!!!!!")

    # cat("\nFinal cv_models check:\n")
    # cat("Number of models:", length(cv_models), "\n")
    # cat("Model names:", names(cv_models), "\n")

    results <- get_cv_metrics(cv_models=cv_models, 
                            all_actual_states=all_actual_states,
                            all_predicted_states=all_predicted_states1,
                            # test_type=Options$clas,
                            features=features
    )



    print_cv_results(results, cv_dir)
    
}



format_cow_results <- function(results) {
    # Format results as a nice table
    results_table <- results
    
    # Calculate percentages for each state
    state_cols <- grep("^correct_|^total_", names(results), value = TRUE)
    state_names <- unique(sub("^(correct_|total_)", "", state_cols))
    
    # Add percentage columns
    for (state in state_names) {
        correct_col <- paste0("correct_", state)
        total_col <- paste0("total_", state)
        pct_col <- paste0("pct_", state)
        
        results_table[[pct_col]] <- ifelse(
            results_table[[total_col]] > 0,
            results_table[[correct_col]] / results_table[[total_col]] * 100,
            NA  # Use NA instead of 0 for no data
        )
    }
    
    # Select and rename columns for display
    display_cols <- c(
        "cow_id",
        "n_labels",
        paste0("pct_", state_names),
        "accuracy"
    )
    
    results_table <- results_table[, display_cols]
    results_table$accuracy <- results_table$accuracy * 100
    
    # Round numeric columns
    numeric_cols <- sapply(results_table, is.numeric)
    results_table[numeric_cols] <- round(results_table[numeric_cols], 2)
    
    return(results_table)
}



# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Required arguments: <util_path> <config_path> <data_path> <output_dir>")
}

util_path <- args[1]
config_path <- args[2]
target_dataset_path <- args[3]
# output_dir <- args[4]
cv_dir <- args[4]
mod_dir <- args[5]
pred_dir <- args[6]

init_the_script(
      util_path=util_path
    , config_path=config_path
    , target_dataset_path=target_dataset_path
    , cv_dir=cv_dir
    , mod_dir=mod_dir
    , pred_dir=pred_dir
    )

quit(status=0)