library(ggplot2)
library(ggridges)
library(dplyr)
library(tidyr)
library(patchwork)



features <- c('step', 'magnitude_mean', 'magnitude_var')

# rb19_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Has_Model_RB_19_20250220_192315/trained_model.rds")
# rb22_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Has_Model_RB_22_20250220_192022/trained_model.rds")
# rb22_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/FinalModels/RB_22_Paper_Model_preds/trained_model.rds")
# combined_model <- readRDS("~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Has_Model_Combined_20250221_120017/trained_model.rds")

# selected_model <- combined_model
# selected_model <- rb22_model
# selected_model <- rb19_model



# valid_states <- selected_model$model$stateNames
valid_states <- c("Resting","Grazing", "Traveling")

# target_dataset <- selected_model$model$data
target_dataset <- read.csv("~/1.Education/CowStudyApp/data/analysis_results/hmm/FinalModels/RB_22_Paper_Model_preds/predictions.csv")

head(target_dataset)
str(target_dataset)
# Define limits for each feature
feature_limits <- list(
  step = c(0, 300),          # Adjust these values
  angle = c(0, 6.28),        # 0 to 2π
  magnitude_mean = c(6, 15),  # Adjust these values
  magnitude_var = c(0, 4)     # Adjust these values
)

feature_bandwidths <- list(
  step = 12,              # Larger bandwidth for step
  magnitude_mean = 0.3,   # Smaller bandwidth for magnitude_mean
  magnitude_var = 0.2     # Smaller bandwidth for magnitude_var
)

# Reshape the data from wide to long format for faceting
plot_data <- target_dataset %>%
  filter(activity %in% valid_states) %>%  # Only keep activities that are in the model states
  select(ID, activity, step, magnitude_mean, magnitude_var) %>%
  pivot_longer(
    cols = features,
    names_to = "feature",
    values_to = "value"
  ) %>%
  # Optional: Remove extreme outliers per feature
  group_by(feature) %>%
  mutate(value = pmin(value, quantile(value, 0.99, na.rm=TRUE))) %>%
  ungroup()




plot_data %>%
  group_by(feature, ID, activity) %>%
  summarise(
    n = n(),
    min_val = min(value, na.rm = TRUE),
    max_val = max(value, na.rm = TRUE),
    na_count = sum(is.na(value))
  ) %>%
  arrange(feature, ID, activity) %>%
  print(n = 30)  # Print first 30 rows to inspect







plot_data$ID <- as.factor(plot_data$ID)
target_dataset$ID <- as.factor(target_dataset$ID)

# Calculate accuracy by cow and activity
accuracy_by_cow_activity <- target_dataset %>%
  group_by(ID, activity) %>%
  summarise(
    accuracy = mean(predicted_state == activity, na.rm = TRUE),
    .groups = "drop"
  )

# Join the accuracy information with the plot data
plot_data <- plot_data %>%
  left_join(accuracy_by_cow_activity, by = c("ID", "activity"))

# Modify the plot creation
plot_list <- lapply(features, function(feat) {
  feature_data <- plot_data %>% 
    filter(feature == feat) %>%
    mutate(ID = factor(ID, levels = sort(unique(ID))),
           # Scale accuracy to alpha range (0.2 to 0.9)
           alpha = scales::rescale(accuracy, to = c(0.2, 0.9)))
  
  ggplot(feature_data, 
         aes(x = value, 
             y = ID,
             group = interaction(ID, activity),
             fill = activity,
             alpha = accuracy)) +  # Use cow-specific accuracy for alpha
    geom_density_ridges(
      scale = 1.2,
      rel_min_height = 0.01,
      bandwidth = feature_bandwidths[[feat]]
    ) +
    facet_wrap(~activity, ncol = 1) +
    theme_ridges(grid = FALSE) +
    scale_x_continuous(limits = feature_limits[[feat]]) +
    scale_alpha_continuous(range = c(0.2, 0.9)) +  # Set alpha range
    labs(x = feat,
         y = "Cow ID") +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 10),
      panel.spacing = unit(1, "lines"),
      plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")
    )
})

# Combine the plots with reduced spacing
combined_plot <- plot_list[[1]] + plot_list[[2]] + plot_list[[3]] +
  plot_layout(ncol = 3, widths = c(1, 0.8, 0.8))

# Add some spacing parameters
combined_plot + 
  # plot_annotation(
  #   title = "Feature Distributions by Cow and Activity",
  #   subtitle = "Opacity indicates prediction accuracy for each cow and activity"
  # ) 
# &
  theme(
    plot.margin = margin(1, 1, 1, 1),
    panel.spacing = unit(0.5, "cm")
  )


# 
# # Modify the plot to adjust bandwidth and handle potential edge cases
# plot_list <- lapply(features, function(feat) {
#   feature_data <- plot_data %>% filter(feature == feat)
#   
#   ggplot(feature_data, aes(x = value, y = ID, fill = activity)) +
#     geom_density_ridges(
#       alpha = 0.6,
#       bandwidth = feature_bandwidths[[feat]],  # Use feature-specific bandwidth
#       scale = 0.9,
#       rel_min_height = 0.01
#     ) +
#     facet_wrap(~activity, ncol = 1) +
#     theme_ridges() +
#     scale_x_continuous(limits = feature_limits[[feat]]) +
#     labs(x = feat,
#          y = "Cow ID") +
#     theme(
#       legend.position = "none",
#       strip.text = element_text(size = 10),
#       panel.spacing = unit(1, "lines")
#     )
# })
# 
# # Combine the plots
# combined_plot <- plot_list[[1]] + plot_list[[2]] + plot_list[[3]] +
#   plot_layout(ncol = 3)
# 
# combined_plot + plot_annotation(
#   title = "Feature Distributions by Cow and Activity"
# )