# run_hmm.R

init_the_script <- function(util_path, config_path, data_path, output_dir){
    # Load dependencies
    cat("Loading dependencies...\n")
    flush.console()
    packages <- c("momentuHMM", "dplyr", "ggplot2", "caret", "fitdistrplus",
                 "circular", "CircStats", "lubridate", 'grid', 'gridExtra',
                 "movMF")


    suppressWarnings({
        suppressPackageStartupMessages({
            invisible(lapply(packages, library, character.only = TRUE))
        })
    })

    # flush.console()
    cat("Sourcing utility functions from:", util_path, "\n")
    source(util_path)
    
    # Load configuration
    cat("Loading configuration from:", config_path, "\n")
    flush.console()
    config <- jsonlite::fromJSON(config_path)
    Options <- config$options
    features <- config$features
    states <- config$states
    set.seed(1)




    all_activities <- c("Grazing", "Resting", "Traveling", "Drinking", "Fighting", "Mineral", "Scratching")


    # dirs <- create_output_structure(base_output_dir=output_dir, features=features)
    dirs <- list(
        base_dir = output_dir,
        plots_dir = file.path(output_dir, "plots"),
        dist_plots_dir = file.path(output_dir, "plots", "distributions")
    )
    
    cat("Loading data from:", data_path, "\n")
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
    target_dataset <- load_csv_data(data_path, features)
    # print(head(target_dataset))
    
    cat("Preparing target data as a momentuHMM object...\n")
    target_dataset <- prepare_hmm_data(data=target_dataset, states=states)
    # print(head(target_dataset))

    # Validate configuration
    # cat("Validating config...\n")
    # validate_config(config, target_dataset)
    
    cat(config$mode)
    if (config$mode == "LOOCV") {
        # Run LOOCV on target dataset
        cat("Starting LOOCV...\n")
        run_LOOCV(config, Options, features, states, dirs, target_dataset, all_activities)
        
    # if(config$mode == "LOOCV"){
    #     run_LOOCV(config, Options, features, states, dirs
    #             , rawData, util_path, config_path
    #             , data_path, output_dir, all_activities)
    }else if(config$mode == "PRODUCT"){
        # str(config$training_info)
        if (!is.null(config$training_info)) {

            cat(sprintf("PRODUCT -- config$training_info$type = %s\n",config$training_info$training_info_type))
            if (config$training_info$training_info_type == "dataset") {
                # Train new model using training dataset
                cat("Configuration set to use 'dataset' as the training type\n")

                cat("Loading the training dataset...\n")
                training_dataset <- load_csv_data(data_path=config$training_info$training_info_path
                                                , features=features, all_activities=all_activities)

                cat("Selecting the labeled data from the training dataset...\n")
                training_dataset <- process_labeled_data(training_dataset, states, all_activities)
                # validate_training_data(training_data)
                
                cat("Creating the training prepData object...\n")
                training_data <- prepare_hmm_data(data=training_data, states=states)

                cat("Computing features from the training data...\n")
                features <- select_best_distributions(data=training_data, 
                                    features=features,
                                    Options=Options,
                                    dirs=dirs)


                cat("Training the model...\n")
                trained_model <- train_model(training_dataset, features, states)

                cat("Applying trained model to target dataset...\n")
                model <- apply_model(target_dataset, trained_model, features, states)
                
                cat("Getting target dataset predictions...\n")
                # get_predictions_from_model(model, target_dataset, dirs, states)
                get_predictions_from_model(model, dirs$base_dir)
                
            } else if (config$training_info$training_info_type == "model") {
                cat("Configuration set to use 'model' as the training type\n")

                # cat("Target Data head:\n")
                # print(head(target_dataset))

                # Load and apply existing model
                cat("Loading model parameters...\n")
                params <- load_model_parameters(config$training_info$training_info_path)
                features <- params$features
                trained_model <- params$model 

                cat("Applying saved model to target dataset...\n")
                model <- apply_model(target_dataset, trained_model, features, states)
                
                cat("Getting target dataset predictions...\n")
                get_predictions_from_model(model, dirs$base_dir)

            } else {
                stop(sprintf("Invalid training_info type `%s`. Must be 'dataset' or 'model'.", config$training_info$type))
            }
        } else {
            # cat("No training info provided. Using the Target data\n")

            cat("Using the target dataset as the training dataset...\n")
            
            cat("Loading the training dataset...\n")
            training_data <- load_csv_data(data_path=data_path,features=features,all_activities=all_activities)

            cat("Selecting the labeled data from the training dataset...\n")
            training_data <- process_labeled_data(raw_data=training_data, states=states, all_activities=all_activities)

            cat("Creating the training prepData object...\n")
            training_data <- prepare_hmm_data(data=training_data, states=states)

            cat("Computing features from the training data...\n")
            features <- select_best_distributions(data=training_data, 
                                features=features,
                                Options=Options,
                                dirs=dirs)

            cat("Training the model...\n")
            trained_model <- train_model(training_data, features, states)

            cat("Applying trained model to target dataset...\n")
            model <- apply_model(target_dataset, trained_model, features, states)
            
            cat("Getting target dataset predictions...\n")
            # get_predictions_from_model(model, target_dataset, dirs, states)
            get_predictions_from_model(model, dirs$base_dir)


            cat("Saving the model and featureset to disk...\n")
            save_model(model,dirs$base_dir, features)

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
        "device_id" = "ID",                    # ID column
        "utm_easting" = "x",                   # Position columns
        "utm_northing" = "y",
        "posix_time" = "posix_time",
        "activity" = "activity"                # Labels
    )

    for (i in 1:nrow(features)) {
        feature_name <- features$name[i]
        column_mapping[[feature_name]] <- feature_name
    }


    # Define features to ignore for validation (prepdata created features)
    ignore_features <- c("step", "angle")
    # print(2)

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

    # print(3)

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

get_predictions_from_model <- function(model, output_dir) {
    # cat("\nDebugging get_predictions_from_model:")
    # cat("\n1. Model structure check:")
    # cat("\n   - States:", length(model$stateNames))
    # cat("\n   - Features:", names(model$mle))
    # cat("\n   - Data rows:", nrow(model$data))
    
    # cat("\n2. MLE parameters:\n")
    # str(model$mle)
    
    # cat("\n3. Attempting viterbi algorithm...")
    tryCatch({
        predictions <- viterbi(model)
        # cat("\n   Viterbi completed successfully")
        # cat("\n   Number of predictions:", length(predictions))
    }, error = function(e) {
        cat("\nError in viterbi:", e$message)
        cat("\nModel state at error:")
        str(model)
        stop(e)
    })


    states <- model$stateNames
    
    # Get the raw data from the model
    results_df <- model$data
    results_df$predicted_state = states[predictions]
    results_df$factored_activity <- NULL
    
    # Create results dataframe

    # results_df <- data.frame(
    #     ID = raw_data$ID,
    #     timestamp = raw_data$posix_time,
    #     actual_states = raw_data$activity,
    #     predicted_state = states[predictions]
    # )
    
    # Save predictions to CSV
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




save_predictions <- function(predictions, target_dataset, base_dir){

    results_df <- data.frame(
            ID = target_dataset$ID,
            posix_time = target_dataset$posix_time,
            actual_activity = target_dataset$activity,
            predicted_activity = predictions
        )
        
        # Save results
        write.csv(results_df, 
                 file.path(base_dir, "predictions.csv"), 
                 row.names = FALSE)
        
}



process_labeled_data <- function(raw_data, states, all_activities) {
    # str(raw_data)

    raw_data$datetime <- as.POSIXct(raw_data$posix_time, 
                                   origin="1970-01-01", 
                                   tz="UTC")
    raw_data$datetime <- with_tz(raw_data$datetime, "America/Denver")
    raw_data$date <- as_date(raw_data$datetime)
    
    filtered_data <- raw_data %>%
        filter(activity %in% all_activities) %>%
        group_by(ID) %>%
        filter(date == min(date)) %>%
        ungroup()
    
    filtered_data$date <- NULL
    filtered_data$datetime <- NULL
    filtered_data$activity <- ifelse(filtered_data$activity %in% states,
                                   filtered_data$activity,
                                   NA)
    
    return(as.data.frame(filtered_data))
}



prepare_hmm_data <- function(data, states) {

    max_magnitude_var = 15
    if ("magnitude_var" %in% names(data)) {
        data$magnitude_var <- pmin(data$magnitude_var, max_magnitude_var)
        
        # Optional: Add log transform to compress the range
        data$magnitude_var <- log1p(data$magnitude_var)  # log1p(x) = log(1 + x)

    }

    prepped_data <- prepData(data)
    prepped_data$factored_activity <- as.integer(factor(data$activity, 
                                                      levels = states))
    return(prepped_data)
} 



train_model <- function(prepped_data, features, states) {
    
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

    model <- fitHMM(data = prepped_data,
                   nbStates = length(states),
                   dist = get_feature_distributions(features),
                   Par0 = parameters,
                   stateNames = states,
                   knownStates = prepped_data$factored_activity,
                   formula = ~1,
                   estAngleMean = list(angle = TRUE))
    
    return(model)
}



apply_model <- function(data, trained_model, features, states) {
    Par0 <- getPar0(trained_model)
    
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

    # Create model without fitting, and copy the trained parameters
    model <- fitHMM(data = data,
                   nbStates = length(states),
                   dist = dist_list,
                   Par0 = modified_Par0,
                   beta0 = Par0$beta,
                   delta0 = Par0$delta,
                   stateNames = states,
                   formula = ~1,
                   estAngleMean = list(angle = TRUE),
                   fit = FALSE)

    # cat("\n3. New model MLE parameters:\n")
    # str(model$mle)
    
    # cat("\n4. Model structure check:")
    # cat("\n   - States:", length(model$stateNames))
    # cat("\n   - Features:", names(model$mle))
    # cat("\n   - Gamma matrix dimensions:", dim(model$mle$gamma))
    # cat("\n   - Delta vector length:", length(model$mle$delta))
    return(model)
}



run_LOOCV <- function(config, Options, features, states, dirs
    , rawData, util_path, config_path
    , data_path, output_dir) {

    cat("Starting HMM analysis...\n")

    # Convert to datetime with proper display format
    rawData$datetime <- tryCatch({
        # Force interpretation as numeric to avoid any string-related issues
        datetime <- as.POSIXct(as.numeric(rawData$posix_time), 
                            origin="1970-01-01", 
                            tz="UTC")
        # Then convert to Denver time
        with_tz(datetime, tzone="America/Denver")
    }, error = function(e) {
        stop("Error converting posix_time to datetime: ", e$message)
    })

    rawData$date <- as_date(rawData$datetime)

    filtered_data <- rawData %>%
        filter(activity %in% all_activities) %>%  # Select rows where activity is in all_activities
        group_by(ID) %>%
        filter(date == min(date)) %>%  # Select rows where date is the first date per ID
        ungroup()

    filtered_data <- as.data.frame(filtered_data)

    filtered_data$date <- NULL
    filtered_data$datetime <- NULL

    filtered_data$activity <- ifelse(filtered_data$activity %in% states, 
                                 filtered_data$activity, 
                                 NA)  # Replace with NA instead of NULL


    # # Validation check
    # if (nrow(filtered_data) > 3000) {
    #     warning("Filtered data has more rows (",nrow(filtered_data),") than expected. Please verify filtering logic.")
    # }

    cat("Sample of input data:\n")
    print(head(filtered_data))
    cat("\nTotal rows: ", nrow(filtered_data))
    cat("\nNumber of unique cows: ", length(unique(filtered_data$ID)), "\n")

    flush.console()

    # Prep HMM Data
    cat("Preparing the momentuHMM object...\n")
    flush.console()

    prepped_data <- prepData(filtered_data)

    prepped_data$factored_activity <- as.integer(
                                factor(
                                prepped_data$activity, 
                                levels = states
                                )
                        )

    cat("Selecting feature distributions...\n")
    flush.console()

    # Select best distributions

    #### Why is it selecting "bad distributions (normal / exp)" for step and mag_var
    #### Fix the issue where weibull uses MLE because it is missing the 'order' parameter of MME
    features <- select_best_distributions(data=prepped_data, 
                                          features=features,
                                          Options=Options,
                                          dirs=dirs)

    cat("Features After Selecting best distributions:\n")
    flush.console()
    for (i in 1:nrow(features)) {
        cat(sprintf("   %s: %s\n", 
                    features$name[i], 
                    ifelse(is.na(features$dist[i]), "not set", features$dist[i])))
        flush.console()
    }


    # Show correlation analysis if requested
    if (Options$show_correlation) {
        correlation_plot <- plot_correlation_matrix(data = prepped_data, 
                                                    features = features)
        save_plot(correlation_plot, 
                 "correlation_matrix", 
                 dirs$plots_dir)
        # Where do we put these plots????
    }

    features$has_zeros <- sapply(features$name, function(fname) {
        any(prepped_data[[fname]] == 0, na.rm = TRUE)
    })

    # cat("Features with zeros:\n")
    # for (i in 1:nrow(features)) {
    #     cat(sprintf("   %s: %s\n", 
    #                 features$name[i], 
    #                 ifelse(features$has_zeros[i], "yes", "no")))
    # }
    # flush.console()

    # Do LOOCV

    cat("Starting LOOCV...\n")
    flush.console()
    state_ids <- 1:length(states)
    all_actual_states <- c()
    all_predicted_states <- c()
    cv_models <- list()


    # str(prepped_data)
    i <- 1
    for (test_cow in levels(prepped_data$ID)) # Doing LOOCV by ID
    {
    cat("+---------------------------------------------------------------------+\n                          LOOCV with test cow", test_cow, "\n")
    flush.console()

    test_data <- subset(prepped_data, ID == test_cow)
    train_data <- subset(prepped_data, ID != test_cow)
    
    # Split data by state
    data_by_state <- list()
    for (i in seq_along(states)) {
        data_by_state[[states[i]]] <- subset(train_data, factored_activity == state_ids[i])
    }
    
    # Calculate parameters for each feature
    parameters <- list()

    for (i in 1:nrow(features)) {
    parameters[[features$name[i]]] <- calculate_parameters(
        data_by_state = data_by_state,
        feature_name = features$name[i],
        dist_type = features$dist[i],
        nbStates = length(states),
        feature_has_zeros = features$has_zeros[i]
    )
    }

    # print(parameters)
    # flush.console()

    # Combine the train and test data (ensure test_data has no activity column for fitting)
    combined_data <- rbind(train_data, test_data)
    # combined_data$activity[is.na(combined_data$activity)] <- NA  # Ensure no activity for test
    
    # cat("Fitting the model...\n")
    # flush.console()
    # Fit the model without known states for the test data (e.g., NA for test part)
    model <- fitHMM(data = combined_data
                    , nbStates = length(states)
                    , dist = get_feature_distributions(features)
                    # , Par0 = get_initial_parameters(features, data_by_state)
                    , Par0 = parameters
                    , stateNames = states
                    # , retryFits = 5
                    , knownStates = c(train_data$factored_activity, rep(NA, nrow(test_data))) # Only training known states
                    , formula = ~1
                    , estAngleMean = list(angle = TRUE) # This needs to be conditionally added if angle is in features.
    )
    

    # print(model$mod)
    # flush.console()

    cat("Running viterbi...\n")
    flush.console()
    predicted_states <- viterbi(model)


    # Now extract the predictions for the test data
    test_predicted_states <- predicted_states[(nrow(train_data) + 1):nrow(combined_data)]
    
    # After predictions, calculate and print detailed statistics for this cow
    valid_indices <- !is.na(test_data$factored_activity)
    test_actual <- test_data$factored_activity[valid_indices]
    test_pred <- test_predicted_states[valid_indices]
    
    # Create a complete confusion matrix with all possible combinations
    cow_cm <- table(factor(states[test_actual], levels=states), 
                    factor(states[test_pred], levels=states))

    cow_accuracy <- mean(test_actual == test_pred)
    log_likelihood <- -model$mod$minimum

    # Print results with proper formatting
    cat("\n========= Results for Cow", test_cow, "=========\n")

    # Print or store the result
    cat(sprintf("Log-likelihood: %.3f\n", log_likelihood))
    cat("Total observations:", length(test_actual), "\n")
    cat("Valid observations:", sum(valid_indices), "\n")
    cat("Accuracy:", round(cow_accuracy * 100, 4), "%\n")

    # Print confusion matrix with proper formatting
    cat("\nConfusion Matrix:\n")
    max_state_length <- max(nchar(states))
    col_width <- max_state_length + 2

    # Print header
    cat(sprintf("%-*s", col_width, "Actual/Pred"))
    cat(paste(sprintf("%*s", col_width, states), collapse = ""))
    cat("\n")

    # Print each row with all columns
    for(i in 1:length(states)) {
        cat(sprintf("%-*s", col_width, states[i]))
        for(j in 1:length(states)) {
            cat(sprintf("%*d", col_width, cow_cm[i,j]))
        }
        cat("\n")
    }
    flush.console()

    
    # Store results for overall analysis
    all_actual_states <- c(all_actual_states, test_data$activity[valid_indices])
    all_predicted_states <- c(all_predicted_states, test_predicted_states[valid_indices])
    
    cv_models[[test_cow]] <- model
    i <- i + 1
    
    }

    cat("Finished LOOCV\n")
    flush.console()
    # Ensure both actual and predicted states are factors with the same levels

    valid_indices <- !is.na(all_actual_states)

    # Filter both vectors using these indices
    all_actual_states <- factor(all_actual_states[valid_indices], levels = states)
    all_predicted_states1 <- factor(states[all_predicted_states[valid_indices]], levels = states)


    results <- get_cv_metrics(cv_models=cv_models, 
                            all_actual_states=all_actual_states,
                            all_predicted_states=all_predicted_states1,
                            # test_type=Options$clas,
                            features=features
    )

    print_cv_results(results)
    flush.console()
}


# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Required arguments: <util_path> <config_path> <data_path> <output_dir>")
}

util_path <- args[1]
config_path <- args[2]
data_path <- args[3]
output_dir <- args[4]


init_the_script(
      util_path=util_path
    , config_path=config_path
    , data_path=data_path
    , output_dir=output_dir
    )

quit(status=0)