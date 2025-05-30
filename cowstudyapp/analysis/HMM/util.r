# # Feature configuration helper functions
get_feature_distributions <- function(features) {
  distributions <- list()
    for (i in 1:nrow(features)) {
        if (features$dist_type[i] != 'covariate')
    distributions[[features$name[i]]] <- features$dist[i]
    }
    return(distributions)
}





# get_initial_parameters <- function(features, data_by_state) {
#   par0_list <- lapply(features, function(feature) {
#     calculate_parameters(data_by_state, feature)
#   })
#   names(par0_list) <- sapply(features, function(f) f$name)
#   return(par0_list)
# }


# create_output_structure <- function(base_output_dir, features) {
#     # Create timestamp
#     timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    
#     # Create feature hash/identifier
#     feature_names <- sort(features$name)  # Sort to ensure consistency
#     feature_id <- paste(feature_names, collapse="_")
#     feature_id <- substr(feature_id, 1, 50)  # Limit length if needed
    
#     # Create directory structure
#     run_dir <- file.path(base_output_dir, feature_id, timestamp)
#     dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
    
#     # Create subdirectories for different plot types
#     plots_dir <- file.path(run_dir, "plots")
#     dist_plots_dir <- file.path(plots_dir, "distributions")
#     dir.create(dist_plots_dir, recursive = TRUE, showWarnings = FALSE)
    
#     # Return the directory structure
#     return(list(
#         base_dir = run_dir,
#         plots_dir = plots_dir,
#         dist_plots_dir = dist_plots_dir
#     ))
# }

# Modified display_plot function
save_plot <- function(plot, filename, directory, width = 10, height = 8) {
    # Clean filename
    clean_name <- gsub("[^[:alnum:]]", "_", filename)
    clean_name <- gsub("_+", "_", clean_name)
    clean_name <- gsub("^_|_$", "", clean_name)
    
    # Create full path
    plot_path <- file.path(directory, paste0(clean_name, ".pdf"))
    
    # Save plot based on its type
    if (is.list(plot) && !is.null(plot$type) && plot$type == "base") {
        pdf(plot_path, width = plot$width, height = plot$height)
        replayPlot(plot$plot)
        dev.off()
    } else if (inherits(plot, "arrangelist") || inherits(plot, "gtable")) {
        pdf(plot_path, width = width, height = height)
        grid.draw(plot)
        dev.off()
    } else if (inherits(plot, "ggplot")) {
        ggsave(plot_path, plot, width = width, height = height)
    } else {
        # For recordPlot objects and other types
        pdf(plot_path, width = width, height = height)
        if (inherits(plot, "recordedplot")) {
            replayPlot(plot)
        } else {
            print(plot)
        }
        dev.off()
    }
    
    cat(sprintf("Saved plot to: %s\n", plot_path))
}



calculate_parameters <- function(data_by_state, feature_name, dist_type, nbStates, feature_has_zeros) {
  
    # cat(sprintf("Feature Name: %s | Dist type: %s, \n", feature_name, dist_type))
    # if (!is.null(dist_type) && dist_type == "covariate") {
    #     return(NULL)  # Covariates don't need distribution parameters
    # }

    validate_numeric <- function(values, name) {
        if (!is.numeric(values)) {
        stop(sprintf("Non-numeric values found in %s", name))
        }
        return(values)
    }
    # print(1)
    calculate_zero_mass <- function(x, feature_name) {
        if (!feature_has_zeros) {
            return(c())
        }
        values <- validate_numeric(x[[feature_name]], feature_name)
        zero_count <- sum(values == 0, na.rm = TRUE)
        total_count <- sum(!is.na(values))
        cat("Zeros (", feature_name, "):", zero_count, "\n")
        if (zero_count > 0) {
            return(rep(zero_count/total_count, nbStates))
        }
        return(rep(0.0001, nbStates))  # Small non-zero value when needed
    }
    
    # print(2)
    #   zero_mass <- if (dist_type %in% c("gamma", "weibull", "exp", "lnorm") && feature_has_zeros) {
    zero_mass <- calculate_zero_mass(do.call(rbind, data_by_state), feature_name)
    




    # #################### Debug
    # if (dist_type == "wrpcauchy") {
    #     cat("\nDebugging Wrapped Cauchy parameters for", feature_name, ":\n")
        
    #     # Print data summary for each state
    #     for (state_name in names(data_by_state)) {
    #         angles <- data_by_state[[state_name]][[feature_name]]
    #         angles <- angles[!is.na(angles)]
            
    #         cat("\nState:", state_name, "\n")
    #         cat("Angle range:", range(angles), "\n")
    #         cat("Mean angle:", mean(angles, na.rm = TRUE), "\n")
    #         cat("Circular mean:", mean.circular(circular(angles)), "\n")
            
    #         # Calculate rho (concentration parameter)
    #         R <- mean(cos(angles)^2 + sin(angles)^2)
    #         cat("Mean resultant length (rho):", R, "\n")
            
    #         # Check if parameters are in valid ranges
    #         if (R >= 1) {
    #             cat("WARNING: rho >= 1, adjusting to 0.99\n")
    #         }
    #         if (R <= 0) {
    #             cat("WARNING: rho <= 0, adjusting to 0.01\n")
    #         }
    #     }
    # }






    # print(3)

    n_states <- length(data_by_state)
    
    params <- switch(dist_type,
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
            
            c(mu, sigma, zero_mass)
          },
        #   "gamma" = {
        #     means <- sapply(data_by_state, function(x) {
        #       values <- validate_numeric(x[[feature_name]], feature_name)
        #       values <- values[values > 0]
        #       mean(values, na.rm = TRUE)
        #     })

        #     sds <- sapply(data_by_state, function(x) {
        #       values <- validate_numeric(x[[feature_name]], feature_name)
        #       values <- values[values > 0]
        #       sd(values, na.rm = TRUE)
        #     })

        #     c(means, sds, zero_mass)
        #     },

          "gamma" = {
            means <- sapply(data_by_state, function(x) {
              values <- validate_numeric(x[[feature_name]], feature_name)
              values <- values[values > 0]
              mean(values, na.rm = TRUE)
            })
            
            vars <- sapply(data_by_state, function(x) {
              values <- validate_numeric(x[[feature_name]], feature_name)
              values <- values[values > 0]
              var(values, na.rm = TRUE)
            })
            
            # Convert mean/variance to shape/scale
            shapes <- means^2 / vars  # shape = mean^2/variance
            scales <- vars / means    # scale = variance/mean
            
            c(shapes, scales, zero_mass)
          },

        #   "wrpcauchy" = {
        #     mu <- sapply(data_by_state, function(x) {
        #       angles <- validate_numeric(x[[feature_name]], feature_name)
        #       mean(angles, na.rm = TRUE)
        #     })
            
        #     rho <- sapply(data_by_state, function(x) {
        #       angles <- validate_numeric(x[[feature_name]], feature_name)
        #       # Calculate concentration parameter (rho) 
        #       # This is a simplified estimate - you might want to use 
        #       # a more sophisticated method for wrapped Cauchy
        #       angles <- angles[!is.na(angles)]
        #       R <- mean(cos(angles)^2 + sin(angles)^2)
        #       R  # R is the mean resultant length
        #     })
            
        #     c(mu, rho)  # No zero mass needed for circular distributions
        #   },


            "wrpcauchy" = {
                mu <- sapply(data_by_state, function(x) {
                    angles <- validate_numeric(x[[feature_name]], feature_name)
                    angles <- angles[!is.na(angles)]
                    # Convert to circular object and get mean direction
                    circ_angles <- circular(angles, units="radians", template="none")
                    mu <- mean.circular(circ_angles)
                    # Ensure mu is in [-pi, pi]
                    mu <- atan2(sin(mu), cos(mu))
                    # cat(sprintf("Calculated mu for state: %.4f\n", mu))
                    mu
                })
                
                rho <- sapply(data_by_state, function(x) {
                    angles <- validate_numeric(x[[feature_name]], feature_name)
                    angles <- angles[!is.na(angles)]
                    # Calculate mean resultant length properly
                    sin_mean <- mean(sin(angles), na.rm = TRUE)
                    cos_mean <- mean(cos(angles), na.rm = TRUE)
                    R <- sqrt(sin_mean^2 + cos_mean^2)
                    # Ensure rho is in (0,1) with some buffer
                    rho <- pmin(pmax(R, 0.01), 0.99)  # Using 0.95 as upper limit for stability
                    # cat(sprintf("Calculated rho for state: %.4f\n", rho))
                    rho
                })
                
                # cat("\nFinal wrapped Cauchy parameters:\n")
                # cat("mu:", sprintf("%.4f ", mu), "\n")
                # cat("rho:", sprintf("%.4f ", rho), "\n")
                
                params <- c(mu, rho)
                if (any(is.na(params))) {
                    stop("NA values in wrapped Cauchy parameters\n",
                        "mu: ", paste(mu, collapse=", "), "\n",
                        "rho: ", paste(rho, collapse=", "))
                }
                
                params
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
                
                c(mu, kappa)  # No zero mass needed for von Mises
            },
            "exp" = {
                rates <- sapply(data_by_state, function(x) {
                values <- validate_numeric(x[[feature_name]], feature_name)
                values <- values[values > 0]
                1/mean(values, na.rm = TRUE)
                })

                c(rates, zero_mass)
            },
            "norm" = {
                means <- sapply(data_by_state, function(x) {
                values <- validate_numeric(x[[feature_name]], feature_name)
                mean(values, na.rm = TRUE)
                })
                
                sds <- sapply(data_by_state, function(x) {
                values <- validate_numeric(x[[feature_name]], feature_name)
                sd(values, na.rm = TRUE)
                })
                
                c(means, sds)  # No zero mass needed for normal
            },
            "weibull" = {
                shapes <- numeric(n_states)
                scales <- numeric(n_states)
                
                for(i in 1:n_states) {
                values <- validate_numeric(data_by_state[[i]][[feature_name]], feature_name)
                values <- values[values > 0]
                mean_val <- mean(values, na.rm = TRUE)
                var_val <- var(values, na.rm = TRUE)
                cv <- sqrt(var_val)/mean_val
                shapes[i] <- (0.9874/cv)^1.0983
                scales[i] <- mean_val/gamma(1 + 1/shapes[i])
                }

                c(shapes, scales, zero_mass)
            },
            stop(sprintf("Unsupported distribution type: %s", dist_type))
  )
  # print(4)
  
  return(params)
}
















calculate_parameters_no_zeromass <- function(data_by_state, feature_name, dist_type) {

  
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

select_best_distributions <- function(data, Options, features, dir) {

    if (Options$show_dist_plots) {
        dist_plots_dir <- file.path(dir, "distributions")
        dir.create(dist_plots_dir, recursive = TRUE, showWarnings = FALSE)
    } else {
        dist_plots_dir <- NULL
    }

    # For each feature
    for (i in 1:nrow(features)) {
        feature_name <- features$name[i]
        dist_type <- features$dist_type[i]
        feature_data <- data[[feature_name]]

        if (dist_type == "covariate") {
            # Skip distribution fitting for covariates
            next
        }


        # Remove NA and infinite values
        valid_indices <- is.finite(feature_data) & !is.na(feature_data)
        feature_data <- feature_data[valid_indices]
        corresponding_activities <- data$activity[valid_indices]  # Get corresponding activities

        cat("Getting fits for ", feature_name, "...\n")
        # flush.console()
        

        # Activity colors
        activity_colors <- c(
            "Grazing" = "#2ecc71",     # Softer green
            "Resting" = "#3498db",     # Nicer blue
            "Traveling" = "#e74c3c",   # Softer red
            "Fighting" = "#f39c12",    # Orange
            "Mineral" = "#9b59b6",     # Purple
            "Drinking" = "#1abc9c",    # Turquoise
            "NA" = "#95a5a6",         # Gray
            "Unknown" = "#bdc3c7"     # Light gray
        )

        distribution_colors <- c(
            "Empirical Density" = "black",
            "vm" = "#E41A1C",
            "wrpcauchy" = "#377EB8",
            "weibull" = "#4DAF4A",
            "lnorm" = "#984EA3",
            "gamma" = "#FF7F00",
            "exp" = "#FFAD33",
            "norm" = "#33B2FF"
        )


        if (dist_type == "circular") {
            # Handle circular data

            fits <- get_circular_fit_metrics(feature_data, Options$distributions$circular)

            # print(fits)
            # Select best distribution based on AIC
            aic_values <- sapply(fits, function(x) x$aic)
            best_dist <- names(which.min(aic_values))
            features$dist[i] <- best_dist
            
            # Plot if requested
            if (Options$show_dist_plots) {
                dist_plot <- plot_circular_distributions(
                      data = feature_data
                    , feature_name = feature_name
                    , fits = fits
                    , Options = Options
                    , activities = corresponding_activities
                    , activity_colors = activity_colors
                    , distribution_colors = distribution_colors    
                )


                save_plot(dist_plot,
                        paste0("distribution_", feature_name),
                        dist_plots_dir)
            }
        }else if (dist_type == "regular")
        {
            # Fit distributions and compare
            fits <- get_regular_fit_metrics(feature_data, Options$distributions$regular)
            
            fit_results <- data.frame(
              Distribution = c("LNORM", "GAMMA", "NORM", "EXP"),
              KS_pvalue = c(
                fits$lnorm$ks_pvalue,
                fits$gamma$ks_pvalue,
                fits$norm$ks_pvalue,
                fits$exp$ks_pvalue
              ),
              KS_statistic = c(
                fits$lnorm$ks_statistic,
                fits$gamma$ks_statistic,
                fits$norm$ks_statistic,
                fits$exp$ks_statistic
              )
            )
            
            cat("\n----------------------- Global Fits -----------------------\n")
            print(fit_results, digits = 4)
            cat("\n-----------------------------------------------------------\n")
            # Select best distribution based on AIC
            aic_values <- sapply(fits, function(x) x$aic)
            best_dist <- names(which.min(aic_values))
            features$dist[i] <- best_dist
            

            if (Options$show_dist_plots) {
                dist_plot <- plot_distribution_analysis(
                    feature_data, 
                    feature_name, 
                    fits, 
                    Options,
                    # activities = data$activity
                    activities = corresponding_activities,
                    activity_colors = activity_colors, distribution_colors = distribution_colors
                )
                save_plot(dist_plot,
                        paste0("distribution_", feature_name),
                        dist_plots_dir)
            }

            # # Plot if requested
            # if (Options$show_dist_plots) {
            #     # dist_plot <- plot_distribution_analysis_old(feature_data, feature_name, fits, Options)
            #     dist_plot <- plot_distribution_analysis(feature_data, feature_name, fits, Options)
            #     save_plot(dist_plot,
            #               paste0("distribution_", feature_name),
            #               dirs$dist_plots_dir)

            #     # cat(sprintf("\nDistribution fits for %s:\n", feature_name))
            #     # print(data.frame(
            #     #     Distribution = names(fits),
            #     #     AIC = sapply(fits, function(x) round(x$aic, 2)),
            #     #     LogLik = sapply(fits, function(x) round(x$loglik, 2))
            #     # ))
            # }
        }else{
          stop(sprintf("Distribution selection failed! Unknown distribution type `%s` for feature `%s`.", dist_type, feature_name))
        }

        # Print summary of the fits once. 
        cat(sprintf("\nDistribution fits for %s:\n", feature_name))
        print(data.frame(
            Distribution = names(fits),
            AIC = sapply(fits, function(x) round(x$aic, 2)),
            LogLik = sapply(fits, function(x) round(x$loglik, 2))
        ))
    }
    
    return(features)
}



get_circular_fit_metrics_old <- function(data, distributions) {
    # Convert to circular data
    angles_rad <- circular(data, units="radians", template="none")
    fits <- list()
    # Von Mises fit
    if ("vm" %in% distributions) {
        tryCatch({
            vm_fit <- mle.vonmises(angles_rad)
            vm_loglik <- sum(dvonmises(angles_rad, 
                          mu = vm_fit$mu, 
                          kappa = vm_fit$kappa, 
                          log = TRUE))

            fits$vm <- list(
                distribution = "vm",
                parameters = list(
                    mu = vm_fit$mu,
                    kappa = vm_fit$kappa
                ),
                loglik = vm_loglik,
                aic = -2 * vm_loglik + 2 * 2,  # 2 parameters
                fit = vm_fit
            )
            # cat("Von Mises fitting finished\n")
        }, error = function(e) {
            cat("Von Mises fitting failed:", e$message, "\n")
        })
    }
    
    # Wrapped Cauchy fit
    if ("wrpcauchy" %in% distributions) {
        tryCatch({
            wc_fit <- mle.wrappedcauchy(angles_rad)
            # Calculate densities first, then take log
            wc_densities <- dwrappedcauchy(angles_rad,
                                         mu = wc_fit$mu,
                                         rho = wc_fit$rho)
            # Add small constant to avoid log(0)
            wc_loglik <- sum(log(wc_densities + .Machine$double.xmin))
            
            fits$wrpcauchy <- list(
                distribution = "wrpcauchy",
                parameters = list(
                    mu = wc_fit$mu,
                    rho = wc_fit$rho
                ),
                loglik = wc_loglik,
                aic = -2 * wc_loglik + 2 * 2,
                fit = wc_fit
            )
            # cat("Wrapped Cauchy fitting finished\n")
        }, error = function(e) {
            cat("Wrapped Cauchy fitting failed:", e$message, "\n")
        })
    }
    
    if (length(fits) == 0) {
        stop("All circular distribution fits failed")
    }
    
    # # Print results summary
    # cat("\nSuccessful fits:\n")
    # for (dist_name in names(fits)) {
    #     cat(dist_name, "- Method:", fits[[dist_name]]$method, 
    #         "AIC:", round(fits[[dist_name]]$aic, 2), "\n")
    # }
    # flush.console()
  
    return(fits)
}








get_circular_fit_metrics <- function(data, distributions) {
    # Transform angles to [0, 2π)
    angles <- as.numeric(data)
    angles[angles < 0] <- angles[angles < 0] + 2*pi
    
    angles_rad <- circular(angles, units="radians", template="none", 
                         type="angles", modulo="2pi")
    
    fits <- list()
    
    # Von Mises fits with different methods
    if ("vm" %in% distributions) {
        tryCatch({
            # Method 1: MLE with raw data
            vm_fit_mle <- mle.vonmises(angles_rad)
            vm_loglik_mle <- sum(dvonmises(angles_rad, 
                                mu = vm_fit_mle$mu, 
                                kappa = vm_fit_mle$kappa, 
                                log = TRUE))
            
            # Method 2: Moment estimation with shifted data
            # Shift data to center the main peak
            density_est <- density.circular(angles_rad, bw=20)
            peak_loc <- density_est$x[which.max(density_est$y)]
            shifted_angles <- (angles_rad - peak_loc) %% (2*pi)
            mu_moment <- mean.circular(shifted_angles) + peak_loc
            R <- rho.circular(shifted_angles)
            kappa_moment <- A1inv(R)
            vm_loglik_moment <- sum(dvonmises(angles_rad, 
                                  mu = mu_moment, 
                                  kappa = kappa_moment, 
                                  log = TRUE))
            
            # Method 3: Robust estimation
            mu_robust <- median.circular(angles_rad)
            # Use circular distance to calculate spread
            diffs <- abs(as.numeric(angles_rad) - as.numeric(mu_robust))
            diffs[diffs > pi] <- 2*pi - diffs[diffs > pi]
            kappa_robust <- 1/(var(diffs) + 0.1)  # Add small constant to prevent extreme values
            vm_loglik_robust <- sum(dvonmises(angles_rad, 
                                  mu = mu_robust, 
                                  kappa = kappa_robust, 
                                  log = TRUE))
            
            # Choose best method
            methods <- list(
                mle = list(mu = vm_fit_mle$mu, kappa = vm_fit_mle$kappa, 
                          loglik = vm_loglik_mle),
                moment = list(mu = mu_moment, kappa = kappa_moment, 
                            loglik = vm_loglik_moment),
                robust = list(mu = mu_robust, kappa = kappa_robust, 
                            loglik = vm_loglik_robust)
            )
            
            best_method <- names(methods)[which.max(sapply(methods, function(x) x$loglik))]
            best_fit <- methods[[best_method]]
            
            fits$vm <- list(
                distribution = "vm",
                parameters = list(
                    mu = best_fit$mu,
                    kappa = best_fit$kappa
                ),
                loglik = best_fit$loglik,
                aic = -2 * best_fit$loglik + 2 * 2,
                method = best_method
            )
        }, error = function(e) {
            cat("Von Mises fitting failed:", e$message, "\n")
        })
    }
    
    # Wrapped Cauchy fits with different methods
    if ("wrpcauchy" %in% distributions) {
        tryCatch({
            # Method 1: MLE with raw data
            wc_fit_mle <- mle.wrappedcauchy(angles_rad)
            wc_densities_mle <- dwrappedcauchy(angles_rad,
                                             mu = wc_fit_mle$mu,
                                             rho = wc_fit_mle$rho)
            wc_loglik_mle <- sum(log(wc_densities_mle + .Machine$double.xmin))
            
            # Method 2: Moment estimation with shifted data
            density_est <- density.circular(angles_rad, bw=20)
            peak_loc <- density_est$x[which.max(density_est$y)]
            shifted_angles <- (angles_rad - peak_loc) %% (2*pi)
            mu_moment <- mean.circular(shifted_angles) + peak_loc
            R <- rho.circular(shifted_angles)
            rho_moment <- R
            wc_densities_moment <- dwrappedcauchy(angles_rad,
                                                mu = mu_moment,
                                                rho = rho_moment)
            wc_loglik_moment <- sum(log(wc_densities_moment + .Machine$double.xmin))
            
            # Choose best method
            methods <- list(
                mle = list(mu = wc_fit_mle$mu, rho = wc_fit_mle$rho, 
                          loglik = wc_loglik_mle),
                moment = list(mu = mu_moment, rho = rho_moment, 
                            loglik = wc_loglik_moment)
            )
            
            best_method <- names(methods)[which.max(sapply(methods, function(x) x$loglik))]
            best_fit <- methods[[best_method]]
            
            fits$wrpcauchy <- list(
                distribution = "wrpcauchy",
                parameters = list(
                    mu = best_fit$mu,
                    rho = best_fit$rho
                ),
                loglik = best_fit$loglik,
                aic = -2 * best_fit$loglik + 2 * 2,
                method = best_method
            )
        }, error = function(e) {
            cat("Wrapped Cauchy fitting failed:", e$message, "\n")
        })
    }
    
    cat("\nFinal comparison:\n")
    for (dist_name in names(fits)) {
        cat(sprintf("%s (%s): AIC=%.2f, loglik=%.2f\n", 
                    dist_name, 
                    fits[[dist_name]]$method,
                    fits[[dist_name]]$aic, 
                    fits[[dist_name]]$loglik))
    }
    return(fits)
}


check_angle_distribution <- function(angles_rad) {
    # Print summary statistics
    cat("Angle Summary:\n")
    cat("Range:", range(angles_rad), "\n")
    cat("Mean:", mean.circular(angles_rad), "\n")
    cat("Median:", median.circular(angles_rad), "\n")
    
    # Print counts in major directions
    breaks <- seq(-pi, pi, length.out=13)  # 12 segments
    hist_counts <- hist(as.numeric(angles_rad), breaks=breaks, plot=FALSE)
    cat("\nCounts by direction:\n")
    for(i in 1:length(hist_counts$counts)) {
        cat(sprintf("%.2f to %.2f: %d\n", 
                   hist_counts$breaks[i], 
                   hist_counts$breaks[i+1],
                   hist_counts$counts[i]))
    }
}


plot_circular_distributions <- function(data, feature_name, fits, Options, 
                                      activities = NULL, 
                                      activity_colors, 
                                      distribution_colors) {
    # Create initial data frame
    plot_data <- data.frame(
        x = as.numeric(data),
        activity = if(!is.null(activities)) activities else rep("Unknown", length(data))
    )
    
    # Calculate bin width and scaling factor
    bin_width <- diff(c(-pi, pi)) / 60  # 60 bins
    n_samples <- length(data)
    scaling_factor <- n_samples * bin_width
    
    # Transform angles
    angles <- plot_data$x
    angles[angles < 0] <- angles[angles < 0] + 2*pi
    
    # Create density estimate
    angles_rad <- circular(angles, units="radians", template="none",
                         type="angles", modulo="2pi")
    dc <- density.circular(angles_rad, bw=40)
    
    # Transform density values and scale them
    dc_x <- dc$x
    dc_x[dc_x > pi] <- dc_x[dc_x > pi] - 2*pi
    sort_idx <- order(dc_x)
    dc_x <- dc_x[sort_idx]
    dc_y <- dc$y[sort_idx] * scaling_factor
    
    # Create base plot
    p1 <- ggplot(plot_data) +
        geom_histogram(aes(x = x, y = after_stat(count), fill = activity),
                      bins = 60, position = "stack") +
        geom_line(data = data.frame(x = dc_x, y = dc_y),
                 aes(x = x, y = y, color = "Empirical Density"),
                 linewidth = 1)
    
    # Add distributions
    theta <- seq(-pi, pi, length=200)
    for (dist_name in names(fits)) {
        fit <- fits[[dist_name]]
        theta_pos <- theta
        theta_pos[theta < 0] <- theta_pos[theta < 0] + 2*pi
        
        density_values <- switch(dist_name,
            "vm" = dvonmises(theta_pos, mu=fit$parameters$mu, kappa=fit$parameters$kappa),
            "wrpcauchy" = dwrappedcauchy(theta_pos, mu=fit$parameters$mu, rho=fit$parameters$rho)
        )
        
        # Scale density values to match histogram height
        density_values <- density_values * scaling_factor
        
        dist_df <- data.frame(x = theta, y = density_values,
                            Distribution = dist_name)
        p1 <- p1 + geom_line(data = dist_df,
                            aes(x = x, y = y, color = Distribution),
                            linewidth = 1)
    }
    
    # Add styling
    p1 <- p1 +
        scale_x_continuous(breaks = c(-pi, -pi/2, 0, pi/2, pi),
                         labels = c("-π", "-π/2", "0", "π/2", "π"),
                         limits = c(-pi, pi)) +
        scale_color_manual(name = "Distributions",
                         values = distribution_colors) +
        scale_fill_manual(name = "Activities",
                        values = activity_colors) +
        theme_minimal() +
        labs(title = paste("Circular Distribution Fitting for", feature_name),
             x = "Angle (radians)",
             y = "Count") +
        theme(legend.position = "right",
              legend.title = element_text(face = "bold"),
              plot.title = element_text(hjust = 0.5),
              legend.box = "vertical")
    
    # Add AIC values to legend
    legend_labels <- sapply(names(fits),
                          function(n) sprintf("%s (AIC=%.2f)", n, fits[[n]]$aic))
    p1 <- p1 + guides(color = guide_legend(title = "Distributions",
                                         labels = legend_labels))
    
    # Add reference lines
    p1 <- p1 +
        geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.3) +
        geom_vline(xintercept = c(-pi, pi), linetype = "dashed", alpha = 0.3)
    
    return(p1)
}



plot_circular_distributions_old <- function(data, feature_name, fits, Options) {
    # First transform angles to [0, 2π)
    angles <- as.numeric(data)
    angles[angles < 0] <- angles[angles < 0] + 2*pi
    
    # Create the density estimate in [0, 2π)
    angles_rad <- circular(angles, units="radians", template="none", 
                         type="angles", modulo="2pi")
    dc <- density.circular(angles_rad, bw=40)  # Adjust bandwidth as needed
    
    # Transform density x values back to [-π, π)
    dc_x <- dc$x
    dc_x[dc_x > pi] <- dc_x[dc_x > pi] - 2*pi
    
    # Sort for proper line plotting
    sort_idx <- order(dc_x)
    dc_x <- dc_x[sort_idx]
    dc_y <- dc$y[sort_idx]
    
    # Create plot
    angle_df <- data.frame(x = as.numeric(data))  # Original data for histogram
    
    p1 <- ggplot() +
        # Histogram
        geom_histogram(data = angle_df,
                      aes(x = x, y = after_stat(density), fill = "Empirical"),
                      bins = 60, alpha = 0.5, boundary = 0) +
        # Density estimate
        geom_line(data = data.frame(x = dc_x, y = dc_y),
                 aes(x = x, y = y, color = "Empirical Density"),
                 linewidth = 1)
    
    # Add fitted distributions
    theta <- seq(-pi, pi, length=200)
    for (dist_name in names(fits)) {
        fit <- fits[[dist_name]]
        # Calculate densities in [0, 2π) and transform back
        theta_pos <- theta
        theta_pos[theta < 0] <- theta_pos[theta < 0] + 2*pi
        
        density_values <- switch(dist_name,
            "vm" = dvonmises(theta_pos, mu=fit$parameters$mu, kappa=fit$parameters$kappa),
            "wrpcauchy" = dwrappedcauchy(theta_pos, mu=fit$parameters$mu, rho=fit$parameters$rho)
        )
        
        dist_df <- data.frame(x = theta, y = density_values,
                            Distribution = dist_name)
        p1 <- p1 + geom_line(data = dist_df,
                            aes(x = x, y = y, color = Distribution),
                            linewidth = 1)
    }
    
    # Add styling
    p1 <- p1 +
        scale_x_continuous(breaks = c(-pi, -pi/2, 0, pi/2, pi),
                         labels = c("-π", "-π/2", "0", "π/2", "π"),
                         limits = c(-pi, pi)) +
        scale_color_manual(name = "Distributions",
                         values = c("Empirical Density" = "black",
                                  "vm" = "#E41A1C",
                                  "wrpcauchy" = "#377EB8")) +
        scale_fill_manual(name = "Data",
                        values = c("Empirical" = "lightgrey")) +
        theme_minimal() +
        labs(title = paste("Circular Distribution Fitting for", feature_name),
             x = "Angle (radians)",
             y = "Density") +
        theme(legend.position = "right",
              legend.title = element_text(face = "bold"),
              plot.title = element_text(hjust = 0.5),
              legend.box = "vertical")
    
    # Add AIC values to legend
    legend_labels <- sapply(names(fits),
                          function(n) sprintf("%s (AIC=%.2f)", n, fits[[n]]$aic))
    
    p1 <- p1 + guides(color = guide_legend(title = "Distributions",
                                         labels = legend_labels))
    
    # Add reference lines
    p1 <- p1 + 
        geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.3) +
        geom_vline(xintercept = c(-pi, pi), linetype = "dashed", alpha = 0.3)
    
    return(p1)
}

get_regular_fit_metrics <- function(data, distributions) {
    if (!requireNamespace("fitdistrplus", quietly = TRUE)) {
        stop("fitdistrplus package is required for distribution fitting")
    }
    # print(data)
    # Data preprocessing
    data <- data[data > 0]  # Ensure positive values
    if (length(data) == 0) {
        stop("No positive values in data after preprocessing")
    }
    
    fits <- list()
    for (distr in distributions) {
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
        
        # Try fitting with different methods if MLE fails
        methods <- c("mme", "mle", "qme")
        for (method in methods) {
            
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

                ks_result <- try({
                    # Create the appropriate distribution function with estimated parameters
                    dist_func <- switch(distr,
                        "lnorm" = function(x) plnorm(x, meanlog = fit$estimate["meanlog"], sdlog = fit$estimate["sdlog"]),
                        "gamma" = function(x) pgamma(x, shape = fit$estimate["shape"], rate = fit$estimate["rate"]),
                        "weibull" = function(x) pweibull(x, shape = fit$estimate["shape"], scale = fit$estimate["scale"]),
                        "norm" = function(x) pnorm(x, mean = fit$estimate["mean"], sd = fit$estimate["sd"]),
                        "exp" = function(x) pexp(x, rate = fit$estimate["rate"])
                    )
                    
                    # Perform the KS test
                    ks.test(data, dist_func)
                }, silent = TRUE)


                if (!inherits(ks_result, "try-error")) {
                    ks_pvalue <- ks_result$p.value
                    ks_statistic <- ks_result$statistic
                } else {
                    ks_pvalue <- NA
                    ks_statistic <- NA
                    cat(" KS test failed for", distr, "\n")
                }



                fits[[distr]] <- list(
                    distribution = distr,
                    aic = AIC(fit),
                    bic = BIC(fit),
                    loglik = fit$loglik,
                    parameters = as.list(fit$estimate),
                    fit = fit,
                    method = method,
                    ks_pvalue = ks_pvalue,
                    ks_statistic = ks_statistic
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
    
    # # Print results summary
    # cat("\nSuccessful fits:\n")
    # for (dist_name in names(fits)) {
    #     cat(dist_name, "- Method:", fits[[dist_name]]$method, 
    #         "AIC:", round(fits[[dist_name]]$aic, 2), "\n")
    # }
    # flush.console()
    
    return(fits)
}



plot_distribution_analysis_old <- function(data, col_name, fits, Options) {
  # Handle outliers if requested
  if (Options$remove_outliers) {
    Q1 <- quantile(data, 0.25)
    Q3 <- quantile(data, 0.75)
    IQR <- Q3 - Q1
    outlier_bounds <- c(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    clean_data <- data[data >= outlier_bounds[1] & data <= outlier_bounds[2]]
    
    # Print outlier information
    cat(sprintf("\nOutliers removed: %d (%.1f%%)\n", 
                sum(data < outlier_bounds[1] | data > outlier_bounds[2]),
                100 * sum(data < outlier_bounds[1] | data > outlier_bounds[2]) / length(data)))
    cat(sprintf("Data range: [%.2f, %.2f] -> [%.2f, %.2f]\n",
                min(data), max(data), min(clean_data), max(clean_data)))
  } else {
    clean_data <- data
  }
  #  print(1)  
  # Print fit metrics table
  metrics_df <- data.frame(
    Distribution = sapply(fits, function(x) x$distribution),
    AIC = sapply(fits, function(x) round(x$aic, 2)),
    BIC = sapply(fits, function(x) round(x$bic, 2)),
    LogLik = sapply(fits, function(x) round(x$loglik, 2))
  )
  #  print(1)
  cat("\nFit Quality Metrics:\n")
  print(metrics_df)
  cat("\n")
  
  # Create base plot data
  x_range <- seq(min(clean_data), max(clean_data), length.out = 200)
  dens_data <- data.frame(x = clean_data)
  
  # Create base plot
  p1 <- ggplot(dens_data, aes(x = x)) +
      geom_histogram(aes(y = after_stat(density), fill = "Empirical"), 
                    bins = 30, alpha = 0.5) +
      geom_density(aes(color = "Empirical Density"), 
                  linewidth = 1)
  
  # Add distribution curves
  for (fit in fits) {
    dist_name <- fit$distribution
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
    
    dist_df <- data.frame(x = x_range, y = y_values, 
                          Distribution = dist_name)
    p1 <- p1 + geom_line(data = dist_df, 
                         aes(x = x, y = y, color = Distribution),
                         linewidth = 1)
  }
  #  print(1)
  # Add scales and theme
  p1 <- p1 + 
    scale_color_manual(name = "Distributions",
                       values = c("Empirical Density" = "black",
                                  "weibull" = "#E41A1C",
                                  "lnorm" = "#377EB8",
                                  "gamma" = "#4DAF4A",
                                  "exp" = "#984EA3",
                                  "norm" = "#FF7F00")) +
    scale_fill_manual(name = "Data",
                      values = c("Empirical" = "lightgrey")) +
    theme_minimal() +
    labs(title = paste("Distribution Fitting for", col_name),
         x = col_name, 
         y = "Density") +
    theme(legend.position = "right",
          legend.title = element_text(face = "bold"),
          plot.title = element_text(hjust = 0.5),
          legend.box = "vertical")
  #  print(1)

  # If showing full range, create a second plot
  if (Options$show_full_range) {
    # Create full range plot
    p2 <- ggplot(data.frame(x = data), aes(x = x)) +
      geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightgrey") +
      theme_minimal() +
      labs(title = "Full Range Distribution",
           x = col_name,
           y = "Density")

    return(gridExtra::grid.arrange(p1, p2, ncol = 1,
                                   heights = c(2, 1)))
  }

  return(p1)
}


plot_distribution_analysis <- function(data, col_name, fits, Options, activities = NULL, distribution_colors, activity_colors) {
    # Create initial data frame with all data and activities
    plot_data <- data.frame(
        x = data,
        activity = if(!is.null(activities)) activities else rep("Unknown", length(data))
    )
    
    # Handle outliers if requested
    if (Options$remove_outliers) {
        Q1 <- quantile(plot_data$x, 0.25)
        Q3 <- quantile(plot_data$x, 0.75)
        IQR <- Q3 - Q1
        outlier_bounds <- c(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        clean_indices <- plot_data$x >= outlier_bounds[1] & plot_data$x <= outlier_bounds[2]
        plot_data <- plot_data[clean_indices, ]
    }
    

  
    # Calculate histogram counts first
    hist_data <- ggplot_build(
        ggplot(plot_data, aes(x = x)) +
        geom_histogram(aes(fill = activity), bins = 30, position = "stack")
    )$data[[1]]
    
    max_count <- max(hist_data$count)
    
    # Calculate bin width for proper density scaling
    bin_width <- diff(range(plot_data$x)) / 30  # 30 bins
    n_samples <- nrow(plot_data)
    scaling_factor <- n_samples * bin_width
    
    # Create base plot with histogram
    p1 <- ggplot(plot_data, aes(x = x)) +
        geom_histogram(aes(y = after_stat(count), fill = activity),
                        bins = 30,
                        position = "stack") +
        scale_y_continuous(name = "Count")
    
    # Add distribution curves scaled to match histogram area
    x_range <- seq(min(plot_data$x), max(plot_data$x), length.out = 200)
    for (fit in fits) {
        dist_name <- fit$distribution
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
        
        # Scale density to match histogram area
        y_values <- y_values * scaling_factor
        
        dist_df <- data.frame(x = x_range, 
                            y = y_values,
                            Distribution = dist_name)
        p1 <- p1 + geom_line(data = dist_df, 
                            aes(x = x, y = y, color = Distribution),
                            linewidth = 1)
    }
    
    # Add empirical density scaled to match
    density_line <- density(plot_data$x)
    density_df <- data.frame(
        x = density_line$x,
        y = density_line$y * scaling_factor
    )
    p1 <- p1 + geom_line(data = density_df,
                        aes(x = x, y = y, color = "Empirical Density"),
                        linewidth = 1)



    # Add scales and theme
    p1 <- p1 + 
        scale_color_manual(name = "Distributions",
                        values = c("Empirical Density" = "black",
                                    "weibull" = "#E41A1C",
                                    "lnorm" = "#377EB8",
                                    "gamma" = "#4DAF4A",
                                    "exp" = "#984EA3",
                                    "norm" = "#FF7F00")) +
        scale_fill_manual(name = "Activities",
                        values = activity_colors) +
        theme_minimal() +
        labs(title = paste("Distribution Fitting for", col_name),
            x = col_name) +
        theme(legend.position = "right",
            legend.title = element_text(face = "bold"),
            plot.title = element_text(hjust = 0.5),
            legend.box = "vertical")

    # If showing full range, create a second plot
    if (Options$show_full_range) {
        full_data <- data.frame(
        x = data,
        activity = if(!is.null(activities)) activities[1:length(data)] 
                    else rep("Unknown", length(data))
        )
        
        # Calculate max count for full range
        full_hist_data <- ggplot_build(
        ggplot(full_data, aes(x = x)) +
            geom_histogram(aes(fill = activity), bins = 30, position = "stack")
        )$data[[1]]
        
        full_max_count <- max(full_hist_data$count)
        
        p2 <- ggplot(full_data, aes(x = x)) +
        geom_histogram(aes(y = after_stat(count),
                            fill = activity),
                        bins = 30,
                        position = "stack") +
        scale_y_continuous(
            name = "Count",
            sec.axis = sec_axis(~./full_max_count, name = "Density")
        ) +
        scale_fill_manual(name = "Activities",
                        values = activity_colors) +
        theme_minimal() +
        labs(title = "Full Range Distribution",
            x = col_name)

        return(gridExtra::grid.arrange(p1, p2, ncol = 1,
                                    heights = c(2, 1)))
    }

    return(p1)
}




get_cv_metrics <- function(cv_models, all_actual_states, all_predicted_states, test_type, features) {
    # Ensure states are factors with same levels
    # all_levels <- unique(c(levels(all_actual_states), levels(all_predicted_states)))
    # all_actual_states <- factor(all_actual_states, levels = all_levels)
    # all_predicted_states <- factor(all_predicted_states, levels = all_levels)
    
    # Overall accuracy and confusion matrix
    accuracy <- mean(all_actual_states == all_predicted_states)
    conf_matrix <- confusionMatrix(all_predicted_states, all_actual_states)
    
    
    # Aggregate transition matrices


    gamma_matrices <- lapply(cv_models, function(m) m$mle$gamma)
    mean_gamma <- Reduce('+', gamma_matrices) / length(gamma_matrices)
    sd_gamma <- sqrt(Reduce('+', lapply(gamma_matrices, function(x) (x - mean_gamma)^2)) / length(gamma_matrices))


    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!bbbbbbbbbbbbbbbbbbbbbbbbbbb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # cat("\nChecking cv_models structure:\n")
    # cat("Number of models:", length(cv_models), "\n")
    # cat("Model names:", names(cv_models), "\n")
    
    # # Check each model's transition matrix
    # cat("\nChecking transition matrices:\n")
    # for(cow_id in names(cv_models)) {
    #     cat("\nCow", cow_id, ":\n")
        
    #     # str(cv_models[[cow_id]]$mle)

    #     if(is.null(cv_models[[cow_id]])) {
    #         cat("- Model is NULL\n")
    #         next
    #     }
    #     if(is.null(cv_models[[cow_id]]$mle)) {
    #         cat("- MLE is NULL\n")
    #         next
    #     }
    #     if(is.null(cv_models[[cow_id]]$mle$gamma)) {
    #         cat("- Gamma matrix is NULL\n")
    #         next
    #     }
    #     cat("- Gamma matrix dimensions:", dim(cv_models[[cow_id]]$mle$gamma), "\n")
    #     print(cv_models[[cow_id]]$mle$gamma)
    # }
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CLOSE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



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



# get_cv_metrics <- function(cv_models, all_actual_states, all_predicted_states, features) {
#     # Ensure states are factors with same levels
#     all_levels <- unique(c(levels(all_actual_states), levels(all_predicted_states)))
#     all_actual_states <- factor(all_actual_states, levels = all_levels)
#     all_predicted_states <- factor(all_predicted_states, levels = all_levels)
    
#     # Overall accuracy and confusion matrix
#     accuracy <- mean(all_actual_states == all_predicted_states)
#     conf_matrix <- confusionMatrix(all_predicted_states, all_actual_states)
    
#     # Aggregate transition matrices - with error checking
#     gamma_matrices <- lapply(cv_models, function(m) {
#         if (is.null(m$mle$gamma)) {
#             cat("Warning: Missing transition matrix in a model\n")
#             return(NULL)
#         }
#         return(m$mle$gamma)
#     })
    
#     # Remove NULL entries and check if we have any matrices
#     gamma_matrices <- gamma_matrices[!sapply(gamma_matrices, is.null)]
    
#     if (length(gamma_matrices) > 0) {
#         mean_gamma <- Reduce('+', gamma_matrices) / length(gamma_matrices)
#         sd_gamma <- sqrt(Reduce('+', lapply(gamma_matrices, function(x) 
#             (x - mean_gamma)^2)) / length(gamma_matrices))
#     } else {
#         cat("Warning: No valid transition matrices found\n")
#         # Create dummy matrices with NAs
#         n_states <- length(all_levels)
#         mean_gamma <- matrix(NA, n_states, n_states)
#         sd_gamma <- matrix(NA, n_states, n_states)
#         rownames(mean_gamma) <- colnames(mean_gamma) <- all_levels
#         rownames(sd_gamma) <- colnames(sd_gamma) <- all_levels
#     }

#     # Rest of the function remains the same...
#     emission_params <- names(cv_models[[1]]$mle)[names(cv_models[[1]]$mle) %in% names(features)]

#   # Process emission parameters dynamically
#   emission_stats <- lapply(emission_params, function(param_name) {
#     params <- lapply(cv_models, function(m) m$mle[[param_name]])
    
    
  
#     if (is.matrix(params[[1]])) {
#       n_states <- nrow(params[[1]])
#       n_components <- ncol(params[[1]])
      
#       means <- matrix(0, n_states, n_components)
#       sds <- matrix(0, n_states, n_components)
      
#       for (i in 1:n_states) {
#         for (j in 1:n_components) {
#           values <- sapply(params, function(p) p[i,j])
#           means[i,j] <- mean(values)
#           sds[i,j] <- sd(values)
#         }
#       }
      
#       list(mean = means, sd = sds)
#     } else {
#       param_matrix <- do.call(rbind, params)
#       list(
#         mean = colMeans(param_matrix),
#         sd = apply(param_matrix, 2, sd)
#       )
#     }
#   })
#   names(emission_stats) <- emission_params
  
#   # Calculate per-state statistics
#   states <- levels(all_actual_states)
#   state_stats <- lapply(states, function(state) {
#     actual_binary <- all_actual_states == state
#     pred_binary <- all_predicted_states == state
    
#     TP <- sum(actual_binary & pred_binary)
#     TN <- sum(!actual_binary & !pred_binary)
#     FP <- sum(!actual_binary & pred_binary)
#     FN <- sum(actual_binary & !pred_binary)
    
#     list(
#       State = state,
#       N_Actual = sum(actual_binary),
#       N_Predicted = sum(pred_binary),
#       Specificity = TN/(TN + FP),
#       Sensitivity = TP/(TP + FN),
#       F1 = 2*TP/(2*TP + FP + FN)
#     )
#   })
#   state_stats_df <- do.call(rbind.data.frame, state_stats)
  
#   return(list(
#     accuracy = accuracy,
#     state_statistics = state_stats_df,
#     confusion_matrix = conf_matrix$table,
#     transition_matrix = list(
#       mean = mean_gamma,
#       sd = sd_gamma
#     ),
#     emission_parameters = emission_stats,
#     features = features
#   ))
# }





print_cv_results <- function(results, cv_dir) {
    if (is.null(results$features)) {
        stop("Features list is required for printing results")
    }

    sink(file.path(cv_dir, "cv_results.txt"))
    
    cat("\n=================================================\n")
    cat("===================== Results ===================\n")
    cat("=================================================\n")
    
    cat("\nOverall Accuracy:", sprintf("%.3f", results$accuracy), "\n")
    
    cat("\nState Statistics:\n")
    # Format state statistics as a readable table
    stats_df <- as.data.frame(results$state_statistics)
    # Round numeric columns
    stats_df[] <- lapply(stats_df, function(x) if(is.numeric(x)) round(x, 3) else x)
    # Print each row manually for better formatting
    for(col in names(stats_df)) {
        cat(sprintf("%-20s: %s\n", col, paste(stats_df[[col]], collapse = " ")))
    }
    
    cat("\nConfusion Matrix:\n")
    print(format(results$confusion_matrix, digits=3))
    
    cat("\nTransition Matrix (mean +/- sd):\n")

    # print(results$transition_matrix)

    # print(results$transition_matrix$mean)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    states <- rownames(results$transition_matrix$mean)
    max_state_length <- max(nchar(states))
    
    # Print transitions
    cat("From/To:", paste(sprintf("%*s", max_state_length, states), collapse = "  "), "\n")
    for(i in 1:nrow(results$transition_matrix$mean)) {
        cat(sprintf("%-*s:  ", max_state_length, states[i]))
        for(j in 1:ncol(results$transition_matrix$mean)) {
            cat(sprintf("%*.3f+/-%.3f  ", 
                max_state_length,
                results$transition_matrix$mean[i,j],
                results$transition_matrix$sd[i,j]))
        }
        cat("\n")
    }
    
    # Distribution parameter names mapping
    dist_param_names <- list(
        "lnorm" = c("Location (mu)", "Scale (sigma)"),
        "vm" = c("Mean (mu)", "Concentration (kappa)"),
        "gamma" = c("Shape (alpha)", "Rate (beta)"),
        "norm" = c("Mean (mu)", "SD (sigma)"),
        "exp" = c("Rate (lambda)"),
        "weibull" = c("Shape (k)", "Scale (lambda)")
    )
    
    # Print emission parameters dynamically
    cat("\nEmission Parameters (mean +/- sd):\n")

    for(param_name in names(results$emission_parameters)) {
        dist_type <- results$features[[param_name]]$dist
        
        cat(sprintf("\n%s (%s distribution):\n", param_name, dist_type))
        
        param_names <- dist_param_names[[dist_type]]
        param_data <- results$emission_parameters[[param_name]]

        for(i in 1:length(param_names)) {
            cat(sprintf("\n%s:\n", param_names[i]))
            for(state in states) {
                state_idx <- which(states == state)
                cat(sprintf("  %-*s: %.3f +/- %.3f\n",
                    max_state_length,
                    state,
                    param_data$mean[i, state_idx],
                    param_data$sd[i, state_idx]))
            }
        }
    }
    sink(NULL)
}