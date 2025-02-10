# library(circular)
# library(CircStats)

# # Feature configuration helper functions
# get_feature_distributions <- function(features) {
#   dist_list <- lapply(features, function(f) f$dist)
#   names(dist_list) <- sapply(features, function(f) f$name)
#   return(dist_list)
# }

# get_initial_parameters <- function(features, data_by_state) {
#   par0_list <- lapply(features, function(feature) {
#     calculate_parameters(data_by_state, feature)
#   })
#   names(par0_list) <- sapply(features, function(f) f$name)
#   return(par0_list)
# }



get_feature_distributions <- function(features) {
    # For a data frame, create named list of distributions
    dist_list <- as.list(features$dist)
    names(dist_list) <- features$name
    return(dist_list)
}

get_initial_parameters <- function(features, data_by_state) {
    # Create list of parameters for each feature
    par0_list <- list()
    for (i in 1:nrow(features)) {
        par0_list[[features$name[i]]] <- calculate_parameters(
            data_by_state = data_by_state,
            feature_name = features$name[i],
            dist_type = features$dist[i]
        )
    }
    return(par0_list)
}




calculate_parameters <- function(data_by_state, feature_name, dist_type) {

  
  validate_numeric <- function(values, name) {
    if (!is.numeric(values)) {
      stop(sprintf("Non-numeric values found in %s", name))
    }
    return(values)
  }
  
  switch(dist_type,
          "lnorm" = {
            mu <- sapply(data_by_state, function(x) {
              values <- validate_numeric(x[[feature_name]], feature_name)
              values <- values[values > 0]
              mean(log(values), na.rm = TRUE)
            })
            
            sigma <- sapply(data_by_state, function(x) {
              values <- validate_numeric(x[[feature_name]], feature_name)
              values <- values[values > 0]
              sd(log(values), na.rm = TRUE)
            })
            return(c(mu, sigma))
          },
          "vm" = {
            mu <- sapply(data_by_state, function(x) {
              angles <- validate_numeric(x[[feature_name]], feature_name)
              mean(angles, na.rm = TRUE)
            })
            
            kappa <- sapply(data_by_state, function(x) {
              angles <- validate_numeric(x[[feature_name]], feature_name)
              angles <- angles[!is.na(angles)]
              est.kappa(circular(angles))
            })
            return(c(mu, kappa))
          },
          "gamma" = {
            shape_rate <- sapply(data_by_state, function(x) {
              values <- validate_numeric(x[[feature_name]], feature_name)
              values <- values[values > 0]
              mean_val <- mean(values, na.rm = TRUE)
              sd_val <- sd(values, na.rm = TRUE)
              return(c(mean_val, sd_val))
            })
            return(shape_rate)
          },
         
          "exp" = {
            # Exponential expects: rate (1/mean), return in a consistent vector format
            rate <- sapply(data_by_state, function(x) {
                values <- validate_numeric(x[[feature_name]], feature_name)
                values <- values[values > 0]
                return(1/mean(values, na.rm = TRUE))  # Keep consistent return format
            })
            return(rate)
          },
          "norm" = {
              # Normal expects: mean, sd, return as a flattened vector
              mean_sd <- sapply(data_by_state, function(x) {
                  values <- validate_numeric(x[[feature_name]], feature_name)
                  c(mean(values, na.rm = TRUE), sd(values, na.rm = TRUE))
              })
              return(c(mean_sd))  # Flattened vector output
          },
          "weibull" = {
              # Weibull expects: shape, scale, return as a flattened vector
              params <- sapply(data_by_state, function(x) {
                  values <- validate_numeric(x[[feature_name]], feature_name)
                  values <- values[values > 0]
                  # Calculate shape and scale from mean and variance
                  mean_val <- mean(values, na.rm = TRUE)
                  var_val <- var(values, na.rm = TRUE)
                  cv <- sqrt(var_val)/mean_val
                  shape <- (0.9874/cv)^1.0983
                  scale <- mean_val/gamma(1 + 1/shape)
                  return(c(shape, scale))  # Flattened vector output
              })
              return(c(params))  # Flattened vector output
          },
          stop(sprintf("Unsupported distribution type: %s", dist_type))
  )
}



plot_correlation_matrix <- function(data, features) {
    # Get feature names from features data frame
    feature_names <- features$name
    
    # Select features from data
    feature_data <- data[, feature_names, drop = FALSE]
    
    # Calculate correlation matrix
    cor_matrix <- cor(feature_data, use = "pairwise.complete.obs")
    
    # Convert to long format
    cor_long <- as.data.frame(as.table(cor_matrix))
    names(cor_long) <- c("Var1", "Var2", "Correlation")
    
    # Create plot
    ggplot(cor_long, aes(Var1, Var2, fill = Correlation)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.2f", Correlation)), 
                  color = ifelse(abs(cor_long$Correlation) < 0.5, "black", "white")) +
        scale_fill_gradient2(
            low = "blue", mid = "white", high = "red",
            midpoint = 0, limits = c(-1, 1)
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = "Feature Correlation Matrix")
}

select_best_distributions <- function(data, Options, features) {
    # Initialize features data frame if not already a data frame
    if (!is.data.frame(features)) {
        features <- data.frame(
            name = sapply(features, function(f) f$name),
            dist = NA_character_,
            stringsAsFactors = FALSE
        )
    }
    
    # For each feature
    for (i in 1:nrow(features)) {
        feature_name <- features$name[i]
        feature_data <- data[[feature_name]]
        
        # Remove NA and infinite values
        feature_data <- feature_data[is.finite(feature_data) & !is.na(feature_data)]
        
        if (feature_name == "angle") {
            # Handle circular data
            features$dist[i] <- "vm"  # von Mises for angles
            next
        }
        cat("Getting fits for ", feature_name, "...\n")
        flush.console()
        
        # Fit distributions and compare
        fits <- get_fit_metrics(feature_data, Options$distributions$regular)
        
        # Select best distribution based on AIC
        aic_values <- sapply(fits, function(x) x$aic)
        best_dist <- names(which.min(aic_values))
        features$dist[i] <- best_dist
        
        # Plot if requested
        if (Options$show_dist_plots) {
            plot_distribution_analysis(feature_data, feature_name, fits, Options)
            cat(sprintf("\nDistribution fits for %s:\n", feature_name))
            print(data.frame(
                Distribution = names(fits),
                AIC = sapply(fits, function(x) round(x$aic, 2)),
                LogLik = sapply(fits, function(x) round(x$loglik, 2))
            ))
        }
    }
    
    return(features)
}


# Helper function for circular distributions
fit_circular_distributions <- function(data) {
  angles_rad <- circular(data, units="radians", template="none")
  
  # Fit von Mises
  vm_fit <- tryCatch({
    mle.vonmises(angles_rad)
  }, error = function(e) NULL)
  
  # Could add other circular distributions here
  # wrapped Cauchy, wrapped normal, etc.
  
  return(list(
    vm = vm_fit
    # Add other fits as needed
  ))
}


# plot_distribution_analysis <- function(data, col_name, fits, Options) {
#   # Handle outliers if requested
#   if (Options$remove_outliers) {
#     Q1 <- quantile(data, 0.25)
#     Q3 <- quantile(data, 0.75)
#     IQR <- Q3 - Q1
#     outlier_bounds <- c(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
#     clean_data <- data[data >= outlier_bounds[1] & data <= outlier_bounds[2]]
    
#     # Print outlier information
#     cat(sprintf("\nOutliers removed: %d (%.1f%%)\n", 
#                 sum(data < outlier_bounds[1] | data > outlier_bounds[2]),
#                 100 * sum(data < outlier_bounds[1] | data > outlier_bounds[2]) / length(data)))
#     cat(sprintf("Data range: [%.2f, %.2f] -> [%.2f, %.2f]\n",
#                 min(data), max(data), min(clean_data), max(clean_data)))
#   } else {
#     clean_data <- data
#   }
#   #  print(1)  
#   # Print fit metrics table
#   metrics_df <- data.frame(
#     Distribution = sapply(fits, function(x) x$distribution),
#     AIC = sapply(fits, function(x) round(x$aic, 2)),
#     BIC = sapply(fits, function(x) round(x$bic, 2)),
#     LogLik = sapply(fits, function(x) round(x$loglik, 2))
#   )
#   #  print(1)
#   cat("\nFit Quality Metrics:\n")
#   print(metrics_df)
#   cat("\n")
  
#   # Create base plot data
#   x_range <- seq(min(clean_data), max(clean_data), length.out = 200)
#   dens_data <- data.frame(x = clean_data)
  

  
  
  
  
#   # Create base plot
#   p1 <- ggplot(dens_data, aes(x = x)) +
#     geom_histogram(aes(y = ..density.., fill = "Empirical"), 
#                    bins = 30, alpha = 0.5) +
#     geom_density(aes(color = "Empirical Density"), 
#                  linewidth = 1)
  
#   # Add distribution curves
#   for (fit in fits) {
#     dist_name <- fit$distribution
#     y_values <- switch(dist_name,
#                        "weibull" = dweibull(x_range, shape = fit$parameters$shape, 
#                                             scale = fit$parameters$scale),
#                        "lnorm" = dlnorm(x_range, meanlog = fit$parameters$meanlog, 
#                                         sdlog = fit$parameters$sdlog),
#                        "gamma" = dgamma(x_range, shape = fit$parameters$shape, 
#                                         rate = fit$parameters$rate),
#                        "exp" = dexp(x_range, rate = fit$parameters$rate),
#                        "norm" = dnorm(x_range, mean = fit$parameters$mean, 
#                                       sd = fit$parameters$sd)
#     )
    
#     dist_df <- data.frame(x = x_range, y = y_values, 
#                           Distribution = dist_name)
#     p1 <- p1 + geom_line(data = dist_df, 
#                          aes(x = x, y = y, color = Distribution),
#                          linewidth = 1)
#   }
#   #  print(1)
#   # Add scales and theme
#   p1 <- p1 + 
#     scale_color_manual(name = "Distributions",
#                        values = c("Empirical Density" = "black",
#                                   "weibull" = "#E41A1C",
#                                   "lnorm" = "#377EB8",
#                                   "gamma" = "#4DAF4A",
#                                   "exp" = "#984EA3",
#                                   "norm" = "#FF7F00")) +
#     scale_fill_manual(name = "Data",
#                       values = c("Empirical" = "lightgrey")) +
#     theme_minimal() +
#     labs(title = paste("Distribution Fitting for", col_name),
#          x = col_name, 
#          y = "Density") +
#     theme(legend.position = "right",
#           legend.title = element_text(face = "bold"),
#           plot.title = element_text(hjust = 0.5),
#           legend.box = "vertical")
#   #  print(1)

#   # If showing full range, create a second plot
#   if (Options$show_full_range) {
#     # Create full range plot
#     p2 <- ggplot(data.frame(x = data), aes(x = x)) +
#       geom_histogram(aes(y = ..density..), bins = 30, fill = "lightgrey") +
#       theme_minimal() +
#       labs(title = "Full Range Distribution",
#            x = col_name,
#            y = "Density")

#     return(gridExtra::grid.arrange(p1, p2, ncol = 1,
#                                    heights = c(2, 1)))
#   }

#   return(p1)
# }


plot_distribution_analysis <- function(data, feature_name, fits, Options) {
    # Handle outliers if requested
    if (Options$remove_outliers) {
        Q1 <- quantile(data, 0.25)
        Q3 <- quantile(data, 0.75)
        IQR <- Q3 - Q1
        outlier_bounds <- c(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        data <- data[data >= outlier_bounds[1] & data <= outlier_bounds[2]]
    }
    
    # Create base plot
    p <- ggplot(data.frame(x = data), aes(x = x)) +
        geom_histogram(aes(y = ..density..), bins = 30, 
                      fill = "lightgrey", color = "black", alpha = 0.5) +
        geom_density(color = "black", linewidth = 1)
    
    # Add fitted distributions
    x_range <- seq(min(data), max(data), length.out = 200)
    colors <- c("lnorm" = "blue", "gamma" = "red", "weibull" = "green",
                "norm" = "purple", "exp" = "orange")
    
    for (dist_name in names(fits)) {
        fit <- fits[[dist_name]]
        y_values <- switch(dist_name,
            "weibull" = dweibull(x_range, shape = fit$parameters$shape, 
                                scale = fit$parameters$scale),
            "lnorm" = dlnorm(x_range, meanlog = fit$parameters$meanlog, 
                            sdlog = fit$parameters$sdlog),
            "gamma" = dgamma(x_range, shape = fit$parameters$shape, 
                            rate = fit$parameters$rate),
            "exp" = dexp(x_range, rate = fit$parameters$rate),
            "norm" = dnorm(x_range, mean = fit$parameters$mean, 
                          sd = fit$parameters$sd)
        )
        
        p <- p + geom_line(data = data.frame(x = x_range, y = y_values),
                          aes(y = y, color = dist_name),
                          linewidth = 1)
    }
    
    p + scale_color_manual(values = colors) +
        theme_minimal() +
        labs(title = sprintf("Distribution Fits for %s", feature_name),
             x = feature_name,
             y = "Density",
             color = "Distribution")
}







get_fit_metrics <- function(data, distributions) {
    if (!requireNamespace("fitdistrplus", quietly = TRUE)) {
        stop("fitdistrplus package is required for distribution fitting")
    }
    
    # Data preprocessing
    data <- data[data > 0]  # Ensure positive values
    if (length(data) == 0) {
        stop("No positive values in data after preprocessing")
    }
    
    # Print data summary for debugging
    # cat("\nData summary before fitting:\n")
    # print(summary(data))
    # flush.console()
    
    fits <- list()
    for (distr in distributions) {
        # cat(" Trying", distr, "distribution...\n")
        # flush.console()
        
        # Calculate better starting values based on data
        start_values <- switch(distr,
            "lnorm" = {
                list(
                    meanlog = mean(log(data)),
                    sdlog = sd(log(data))
                )
            },
            "gamma" = {
                m <- mean(data)
                v <- var(data)
                list(
                    shape = (m^2)/v,
                    rate = m/v
                )
            },
            "weibull" = {
                # Method of moments estimates for Weibull
                mean_val <- mean(data)
                var_val <- var(data)
                cv <- sqrt(var_val)/mean_val
                shape_approx <- (0.9874/cv)^1.0983
                scale_approx <- mean_val/gamma(1 + 1/shape_approx)
                list(
                    shape = shape_approx,
                    scale = scale_approx
                )
            },
            "norm" = {
                # Normal can handle any values
                list(
                    mean = mean(data),
                    sd = sd(data)
                )
            },
            "exp" = {
                # Need positive values for exponential
                pos_data <- data[data > 0]
                if (length(pos_data) == 0) {
                    cat(" Skipping exponential - no positive values\n")
                    next
                }
                list(
                    rate = 1/mean(pos_data)
                )
            }
        )
        
        # Print starting values for debugging
        # cat(" Starting values for", distr, ":\n")
        # print(start_values)
        # flush.console()
        
        # Try fitting with different methods if MLE fails
        methods <- c("mme", "mle", "qme")
        for (method in methods) {
            # cat(" Attempting fit with method:", method, "\n")
            # flush.console()
            
            fit <- try(fitdistrplus::fitdist(
                data,
                distr,
                method = method,
                start = start_values,
                lower = c(0.001, 0.001),  # Prevent parameters from getting too close to 0
                control = list(
                    maxit = 1000,
                    reltol = 1e-8
                )
            ), silent = TRUE)
            
            if (!inherits(fit, "try-error")) {
                # cat(" Successful fit with method:", method, "\n")
                fits[[distr]] <- list(
                    distribution = distr,
                    aic = AIC(fit),
                    bic = BIC(fit),
                    loglik = fit$loglik,
                    parameters = as.list(fit$estimate),
                    fit = fit,
                    method = method
                )
                break  # Stop trying other methods if one succeeds
            } else {
                cat(" Failed with method:", method, "\n")
                cat(" Error:", attr(fit, "condition")$message, "\n")
            }
            flush.console()
        }
    }
    
    if (length(fits) == 0) {
        stop("All distribution fits failed")
    }
    
    # Print results summary
    cat("\nSuccessful fits:\n")
    for (dist_name in names(fits)) {
        cat(dist_name, "- Method:", fits[[dist_name]]$method, 
            "AIC:", round(fits[[dist_name]]$aic, 2), "\n")
    }
    flush.console()
    
    return(fits)
}

get_cv_metrics <- function(cv_models, all_actual_states, all_predicted_states, test_type, features) {
  # Ensure states are factors with same levels
  all_levels <- unique(c(levels(all_actual_states), levels(all_predicted_states)))
  all_actual_states <- factor(all_actual_states, levels = all_levels)
  all_predicted_states <- factor(all_predicted_states, levels = all_levels)
  
  # Overall accuracy and confusion matrix
  accuracy <- mean(all_actual_states == all_predicted_states)
  conf_matrix <- confusionMatrix(all_predicted_states, all_actual_states)
  
  
  # Aggregate transition matrices
  gamma_matrices <- lapply(cv_models, function(m) m$mle$gamma)
  mean_gamma <- Reduce('+', gamma_matrices) / length(gamma_matrices)
  sd_gamma <- sqrt(Reduce('+', lapply(gamma_matrices, function(x) (x - mean_gamma)^2)) / length(gamma_matrices))
  
  emission_params <- names(cv_models[[1]]$mle)[names(cv_models[[1]]$mle) %in% names(features)]
  
  # Process emission parameters dynamically
  emission_stats <- lapply(emission_params, function(param_name) {
    params <- lapply(cv_models, function(m) m$mle[[param_name]])
    
    
  
    if (is.matrix(params[[1]])) {
      n_states <- nrow(params[[1]])
      n_components <- ncol(params[[1]])
      
      means <- matrix(0, n_states, n_components)
      sds <- matrix(0, n_states, n_components)
      
      for (i in 1:n_states) {
        for (j in 1:n_components) {
          values <- sapply(params, function(p) p[i,j])
          means[i,j] <- mean(values)
          sds[i,j] <- sd(values)
        }
      }
      
      list(mean = means, sd = sds)
    } else {
      param_matrix <- do.call(rbind, params)
      list(
        mean = colMeans(param_matrix),
        sd = apply(param_matrix, 2, sd)
      )
    }
  })
  names(emission_stats) <- emission_params
  
  # Calculate per-state statistics
  states <- levels(all_actual_states)
  state_stats <- lapply(states, function(state) {
    actual_binary <- all_actual_states == state
    pred_binary <- all_predicted_states == state
    
    TP <- sum(actual_binary & pred_binary)
    TN <- sum(!actual_binary & !pred_binary)
    FP <- sum(!actual_binary & pred_binary)
    FN <- sum(actual_binary & !pred_binary)
    
    list(
      State = state,
      N_Actual = sum(actual_binary),
      N_Predicted = sum(pred_binary),
      Specificity = TN/(TN + FP),
      Sensitivity = TP/(TP + FN),
      F1 = 2*TP/(2*TP + FP + FN)
    )
  })
  state_stats_df <- do.call(rbind.data.frame, state_stats)
  
  return(list(
    # test_type = test_type,
    accuracy = accuracy,
    state_statistics = state_stats_df,
    confusion_matrix = conf_matrix$table,
    transition_matrix = list(
      mean = mean_gamma,
      sd = sd_gamma
    ),
    emission_parameters = emission_stats,
    features = features
  ))
}



print_cv_results <- function(results) {
    if (is.null(results$features)) {
        stop("Features list is required for printing results")
    }
    
    cat("\n=================================================\n")
    # cat("Test Type:", results$test_type, "\n")
    # cat("=================================================\n")
    
    cat("\nOverall Accuracy:", sprintf("%.3f", results$accuracy), "\n")
    
    cat("\nState Statistics:\n")
    print(data.frame(lapply(results$state_statistics, function(x) 
        if(is.numeric(x)) round(x, 3) else x)))
    
    cat("\nConfusion Matrix:\n")
    print(results$confusion_matrix)
    
    cat("\nTransition Matrix (mean ± sd):\n")
    states <- rownames(results$transition_matrix$mean)
    max_state_length <- max(nchar(states))
    
    # Print transitions
    cat("From/To:", paste(sprintf("%*s", max_state_length, states), collapse = "  "), "\n")
    for(i in 1:nrow(results$transition_matrix$mean)) {
        cat(sprintf("%-*s:  ", max_state_length, states[i]))
        for(j in 1:ncol(results$transition_matrix$mean)) {
            cat(sprintf("%*.3f±%.3f  ", 
                max_state_length,
                results$transition_matrix$mean[i,j],
                results$transition_matrix$sd[i,j]))
        }
        cat("\n")
    }
    
    # Distribution parameter names mapping
    dist_param_names <- list(
        "lnorm" = c("Location (μ)", "Scale (σ)"),
        "vm" = c("Mean (μ)", "Concentration (κ)"),
        "gamma" = c("Shape (α)", "Rate (β)"),
        "norm" = c("Mean (μ)", "SD (σ)"),
        "exp" = c("Rate (λ)"),
        "weibull" = c("Shape (k)", "Scale (λ)")
    )
    
    # Print emission parameters dynamically
    cat("\nEmission Parameters (mean ± sd):\n")

    for(param_name in names(results$emission_parameters)) {
        dist_type <- results$features[[param_name]]$dist

        print(param_name)
        cat(sprintf("\n%s (%s distribution):\n", param_name, dist_type))
        
        param_names <- dist_param_names[[dist_type]]
        
        param_data <- results$emission_parameters[[param_name]]

        for(i in 1:length(param_names)) {
            cat(sprintf("\n%s:\n", param_names[i]))
            for(state in states) {
                state_idx <- which(states == state)
                cat(sprintf("  %-*s: %.3f ± %.3f\n",
                    max_state_length,
                    state,
                    param_data$mean[i, state_idx],
                    param_data$sd[i, state_idx]))
            }
        }
    }
}



post_predictions_to_db <- function(prepped_data, decoded_states, states, table_name, overwrite = TRUE) {
  # Get all feature names from prepped_data excluding ID, posix_time, and activity
  feature_names <- setdiff(names(prepped_data), 
                           c("ID", "posix_time", "activity", "factored_activity"))
  
  # Dynamically create output data frame
  output_data <- data.frame(
    collar_id = prepped_data$ID,
    posix_time = prepped_data$posix_time,
    activity = states[decoded_states]
  )
  
  # Add all features to output data
  for(feature in feature_names) {
    output_data[[feature]] <- prepped_data[[feature]]
  }
  
  tryCatch({
    con <- get_db_connection()
    
    # Dynamically create table columns based on features
    feature_columns <- paste(
      sapply(feature_names, function(f) sprintf("%s DOUBLE", f)),
      collapse = ",\n"
    )
    
    create_table_query <- sprintf("
            CREATE TABLE IF NOT EXISTS %s (
                collar_id VARCHAR(50) NOT NULL,
                posix_time BIGINT NOT NULL,
                %s,
                activity VARCHAR(50) NOT NULL,
                PRIMARY KEY (collar_id, posix_time)
            )", table_name, feature_columns)
    
    dbExecute(con, create_table_query)
    
    if (overwrite) {
      dbExecute(con, sprintf("TRUNCATE TABLE %s", table_name))
    }
    
    dbWriteTable(
      con, 
      table_name, 
      output_data,
      append = TRUE,
      row.names = FALSE
    )
    
    cat("\nPrediction Summary:\n")
    print(table(states[decoded_states]))
    
    dbDisconnect(con)
    
    cat(sprintf("\nSuccessfully wrote %d rows to table '%s'\n", 
                nrow(output_data), 
                table_name))
    
  }, error = function(e) {
    cat(sprintf("\nError writing to database: %s\n", e$message))
    if(exists('con') && dbIsValid(con)) {
      dbDisconnect(con)
    }
    stop(e)
  })
  
  return(output_data)
}