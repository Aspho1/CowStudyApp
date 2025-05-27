library(momentuHMM)

# Function to calculate statistics for a dataset and state
calculate_stats <- function(data, columns, state = NULL) {
  if (!is.null(state)) {
    data <- data[data$activity == state, ]
  }
  
  result <- c()
  for (col in columns) {
    result <- c(result,
                nrow(data),  # n
                min(data[[col]], na.rm = TRUE),
                mean(data[[col]], na.rm = TRUE),
                median(data[[col]], na.rm = TRUE),
                max(data[[col]], na.rm = TRUE),
                sd(data[[col]], na.rm = TRUE)
    )
  }
  return(result)
}

# Load the models
rb19_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Has_Model_RB_19_20250220_192315/trained_model.rds")
rb22_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Has_Model_RB_22_20250220_192022/trained_model.rds")

# Columns to analyze
columns <- c("step", "angle", "magnitude_mean", "magnitude_var")
states <- c("Grazing", "Resting", "Traveling")

# Initialize results matrix
results <- matrix(nrow = 12, ncol = 16)  # 2 models x 6 stats = 12 rows, 4 categories x 4 measures = 16 cols

# Calculate stats for each dataset
row_idx <- 1
for (model_name in c("RB_19", "RB_22")) {
  model_data <- if(model_name == "RB_19") rb19_model$model$data else rb22_model$model$data
  
  # Filter out NA activities for global stats
  global_data <- model_data[model_data$activity != "", ]
  
  # For each statistic type (n, min, mean, median, max, sd)
  for (stat_idx in 1:6) {
    row <- c()
    
    # Global stats (with filtered data)
    row <- c(row, calculate_stats(global_data, columns))
    
    # State-specific stats
    for (state in states) {
      row <- c(row, calculate_stats(model_data, columns, state))
    }
    
    results[row_idx, ] <- row[stat_idx + seq(0, length(row)-1, by=6)]
    row_idx <- row_idx + 1
  }
}

# Write results with no headers
write.table(results, "hmm_statistics.csv", sep=",", row.names=FALSE, col.names=FALSE)

# Print to console for verification
print(results)


