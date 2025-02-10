# run_hmm.R
run_analysis <- function(util_path, config_path, data_path, output_dir) {
    
    # Sys.setlocale("LC_ALL", "en_US.UTF-8")

    cat("Starting HMM analysis...\n")
    flush.console()
    
    # Load dependencies
    cat("Loading dependencies...\n")
    flush.console()
    packages <- c("momentuHMM", "dplyr", "ggplot2", "caret", "fitdistrplus",
                 "circular", "CircStats", "lubridate")
                 
    suppressWarnings({
        suppressPackageStartupMessages({
            invisible(lapply(packages, library, character.only = TRUE))
        })
    })

    cat("Sourcing utility functions from:", util_path, "\n")
    flush.console()
    source(util_path)
    
    # Load configuration
    cat("Loading configuration from:", config_path, "\n")
    flush.console()
    config <- jsonlite::fromJSON(config_path)
    Options <- config$options
    features <- config$features
    states <- config$states
    set.seed(1)

    
    cat("Loading data from:", data_path, "\n")
    flush.console()

    cat("States:", paste(states, collapse=", "), "\n")

    cat("Features:", features$name, "\n")

    flush.console()
    if (!is.null(Options)) {
        cat("\nOptions:\n")
        for (name in names(Options)) {
            if (is.list(Options[[name]])) {
                # If it's a list, print each element separately
                cat("- ", name, ":\n", sep="")
                for (subname in names(Options[[name]])) {
                    # Handle nested lists (like distributions)
                    if (is.list(Options[[name]][[subname]]) || is.vector(Options[[name]][[subname]]) && length(Options[[name]][[subname]]) > 1) {
                        cat("  - ", subname, ": [", sep="")
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
        flush.console()
    }

    flush.console()
    

    # Load, rename, and select data
    rawData <- read.csv(data_path)


    # Column mapping
    cat("Selecting and renaming data...\n")
    flush.console()
    column_mapping <- list(
        "device_id" = "ID",                    # ID column
        "utm_easting" = "x",                   # Position columns
        "utm_northing" = "y",
        "magnitude_mean" = "mag_mean",     # Accelerometer features
        "magnitude_var" = "mag_var",
        "posix_time" = "posix_time",
        "activity" = "activity"                # Labels
    )

    all_activities <- c("Grazing", "Resting", "Traveling", "Drinking", "Fighting", "Mineral", "Scratching")

    # Rename columns
    for (old_name in names(column_mapping)) {
        if (old_name %in% names(rawData)) {
            names(rawData)[names(rawData) == old_name] <- column_mapping[[old_name]]
        } else {
            warning(sprintf("Column '%s' not found in data", old_name))
        }
    }

    # Select only the columns we need
    required_columns <- unlist(column_mapping)
    missing_columns <- required_columns[!required_columns %in% names(rawData)]
    if (length(missing_columns) > 0) {
        stop(sprintf("Missing required columns: %s", 
                    paste(missing_columns, collapse=", ")))
    }
    rawData <- rawData[, required_columns]
    rawData$ID <- as.integer(rawData$ID)

    # cat("Converting timestamps...\n")

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

    print(head(filtered_data))
    flush.console()

    filtered_data$date <- NULL
    filtered_data$datetime <- NULL

    # states
    # If filtered_data$activity is in states, keep it the value in activity.
    # Else, replace the value with NULL
    filtered_data$activity <- ifelse(filtered_data$activity %in% states, 
                                 filtered_data$activity, 
                                 NA)  # Replace with NA instead of NULL


    # Validation check
    if (nrow(filtered_data) > 3000) {
        warning("Filtered data has more rows (",nrow(filtered_data),") than expected. Please verify filtering logic.")
    }

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

    # print(head(prepped_data))

    # cat("Summary of the momentuHMM object: \n", summary(prepped_data),"\n")
    # flush.console()

    cat("Features before SBD:\n")
    flush.console()
    for (i in 1:nrow(features)) {
        cat(sprintf("%s: %s\n", 
                    features$name[i], 
                    ifelse(is.na(features$dist[i]), "not set", features$dist[i])))
        flush.console()
    }

    cat("Selecting feature distributions...\n")
    flush.console()

    # Select best distributions

    #### Why is it selecting "bad distributions (normal / exp)" for step and mag_var
    features <- select_best_distributions(data=prepped_data, 
                                          features=features,
                                          Options=Options)
    cat("Features After SBD:\n")
    flush.console()
    for (i in 1:nrow(features)) {
        cat(sprintf("%s: %s\n", 
                    features$name[i], 
                    ifelse(is.na(features$dist[i]), "not set", features$dist[i])))
        flush.console()
    }


    # Show correlation analysis if requested
    if (Options$show_correlation) {
        correlation_plot <- plot_correlation_matrix(data = prepped_data, 
                                                    features = features)
        print(correlation_plot)

        # Where do we put these plots????
    }


    # Do LOOCV

    cat("Starting LOOCV...\n")
    flush.console()
    state_ids <- 1:length(states)
    all_actual_states <- c()
    all_predicted_states <- c()
    cv_models <- list()

    str(prepped_data)
    i <- 1
    for (test_cow in unique(prepped_data$ID)) # Doing LOOCV by ID
    {

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
        parameters[[features$name[i]]] <- calculate_parameters(data_by_state=data_by_state,
                                                               feature_name=features$name[i],
                                                               dist_type=features$dist[i]
         )
    }
    # print(parameters)
    cat("Initial parameters for test cow ", test_cow, "\n")
    flush.console()

    # Combine the train and test data (ensure test_data has no activity column for fitting)
    combined_data <- rbind(train_data, test_data)
    # combined_data$activity[is.na(combined_data$activity)] <- NA  # Ensure no activity for test
    
    cat("Fitting the model...\n")
    flush.console()
    # Fit the model without known states for the test data (e.g., NA for test part)
    model <- fitHMM(data = combined_data
                    , nbStates = length(states)
                    , dist = get_feature_distributions(features)
                    , Par0 = get_initial_parameters(features, data_by_state)
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
    
    # Calculate statistics for this cow
    cow_cm <- table(Actual = states[test_actual], Predicted = states[test_pred])
    cow_accuracy <- mean(test_actual == test_pred)
    
    # Print detailed results for this cow
    cat("\n========= Results for Cow", test_cow, "=========\n")
    flush.console()
    cat("Total observations:", length(test_actual), "\n")
    flush.console()
    cat("Valid observations:", sum(valid_indices), "\n")
    flush.console()
    cat("Accuracy:", round(cow_accuracy * 100, 4), "%\n")
    flush.console()
    cat("\nConfusion Matrix(states=",paste(states, collapse=", "),"):\n", cow_cm, "\n")
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

    cat(output_dir)
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

cat("Running analysis with:\n")
cat("Util path:", util_path, "\n")
cat("Config path:", config_path, "\n")
cat("Data path:", data_path, "\n")
cat("Output directory:", output_dir, "\n")
flush.console()

run_analysis(util_path=util_path, config_path=config_path, data_path=data_path, output_dir=output_dir)