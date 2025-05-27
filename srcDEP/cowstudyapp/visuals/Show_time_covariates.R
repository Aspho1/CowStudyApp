library(ggplot2)
library(lubridate)
library(dplyr)

create_time_covariate <- function(posix_time, latitude, longitude, date, timezone = "America/Denver") {
  # Convert timestamp to datetime while preserving the actual date
  datetime <- as.POSIXct(posix_time, origin="1970-01-01", tz=timezone)
  current_date <- as.Date(date)
  
  # Create data frame for sun times calculation
  day_data <- data.frame(
    date = current_date,
    lat = latitude[1],
    lon = longitude[1]
  )
  
  # Get sun times for this specific date
  sun_times <- getSunlightTimes(
    data = day_data, 
    keep = c("sunrise", "sunset"), 
    tz = timezone
  )
  
  # cat(sprintf("Processing date: %s\n", format(current_date, "%Y-%m-%d")))
  
  # Convert times to hours, using the actual sunrise/sunset times
  hours <- as.numeric(format(datetime, "%H")) + 
           as.numeric(format(datetime, "%M"))/60
  
  sunrise_time <- as.POSIXct(sun_times$sunrise)
  sunset_time <- as.POSIXct(sun_times$sunset)
  
  sunrise_hours <- as.numeric(format(sunrise_time, "%H")) +
                   as.numeric(format(sunrise_time, "%M"))/60
  
  sunset_hours <- as.numeric(format(sunset_time, "%H")) +
                  as.numeric(format(sunset_time, "%M"))/60
  
  # Calculate peak times
  morning_peak_time <- sunrise_hours + 1.5
  evening_peak_time <- sunset_hours - 1.5
  
  # Debug print
  # cat(sprintf("Sunrise: %.2f, Sunset: %.2f, Morning Peak: %.2f, Evening Peak: %.2f\n",
  #             sunrise_hours, sunset_hours,
  #             morning_peak_time, evening_peak_time))
  
  # Calculate mu values for von Mises distributions
  mu1 <- (morning_peak_time / 24) * 2 * pi
  mu2 <- (evening_peak_time / 24) * 2 * pi
  day_center <- (mu1 + mu2) / 2
  
  time_between_peaks <- mu2 - mu1
  
  # Normalize time to circular scale
  time_normalized <- (hours / 24) * 2 * pi
  
  # Calculate peaks
  peak1 <- dvonmises(circular(time_normalized), mu = mu1, kappa = 12/time_between_peaks)
  peak2 <- dvonmises(circular(time_normalized), mu = mu2, kappa = 12/time_between_peaks)
  peak3 <- 0.5 * dvonmises(circular(time_normalized), mu = day_center, kappa = time_between_peaks/15)
  
  # Combine peaks
  diurnal_score <- ((peak1 + peak2) / 2) + peak3
  
  # Normalize
  final_score <- diurnal_score / max(diurnal_score, na.rm = TRUE)
  
  return(final_score)
}


test_annual_covariate <- function() {
  
  
  winter_solstice_2021 <- as.Date("2021-12-21")
  spring_equinox_2022 <- as.Date("2022-03-20")
  summer_solstice_2022 <- as.Date("2022-06-21")
  fall_equinox_2022 <- as.Date("2022-09-22")
  winter_solstice_2022 <- as.Date("2022-12-21")
  
  # Create sequence of dates (every 5 days)
  all_dates <- seq.Date(
    from = winter_solstice_2021,
    to = winter_solstice_2022,
    by = "1 days"
  )
  
  special_dates <- c(winter_solstice_2021, spring_equinox_2022, 
                     summer_solstice_2022, fall_equinox_2022, 
                     winter_solstice_2022)
  all_dates <- unique(sort(c(all_dates, special_dates)))
  

  lat <- 45.5767117
  lon <- -111.6349473
  
  # Create results list
  results <- list()
  
  for(date in all_dates) {

    times <- seq(
      from = as_datetime(date, tz = "America/Denver"),
      to = as_datetime(date, tz = "America/Denver") + hours(23) + minutes(45),
      by = "15 min"
    )
      
    # Calculate covariate
    scores <- create_time_covariate(
      posix_time = as.numeric(times),
      latitude = rep(lat, length(times)),
      longitude = rep(lon, length(times)),
      date = date
    )
    

    results[[as.character(date)]] <- data.frame(
      datetime = times,
      score = scores,
      date = date,
      hour = hour(times) + minute(times)/60,
      is_special = date %in% special_dates,
      special_type = case_when(
        date == winter_solstice_2021 ~ "Winter Solstice 2021",
        date == spring_equinox_2022 ~ "Spring Equinox",
        date == summer_solstice_2022 ~ "Summer Solstice",
        date == fall_equinox_2022 ~ "Fall Equinox",
        date == winter_solstice_2022 ~ "Winter Solstice 2022",
        TRUE ~ "Regular Day"
      )
    )
  }
  
  
  
  # Combine all results
  all_results <- do.call(rbind, results)
  
  # Format dates for better legend and convert to numeric for color scaling
  all_results$date_numeric <- as.numeric(all_results$date)

  # Add these diagnostic checks before plotting
  print("Number of unique dates:")
  print(length(unique(all_results$date)))
  
  # Check peak times for a few dates
  print("Sample of peak times by date:")
  peak_times <- tapply(all_results$score, all_results$date, function(x) {
    peak_hour <- all_results$hour[which.max(x)]
    return(peak_hour)
  })
  print(head(peak_times))
  
  
  
  
  # Try a different visualization approach
  p <- ggplot(all_results, aes(x = hour, y = score, 
                               group = date, 
                               color = date_numeric)) +
    # Draw regular days first
    geom_line(data = subset(all_results, !is_special),
              linewidth = 0.1,
              alpha = 0.1) +
    # Draw special days on top
    geom_line(data = subset(all_results, is_special),
              linewidth = 2,
              alpha = 1) +
    scale_x_continuous(
      breaks = 0:23,
      labels = sprintf("%02d:00", 0:23),
      limits = c(0, 23),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, by = 0.1),
      labels = sprintf("%.1f", seq(0, 1, by = 0.1)),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    scale_color_gradientn(
      colors = c("#0000FF", "#FF0000", "#0000FF"),
      labels = function(x) format(as.Date(origin = "1970-01-01", x), "%b %d"),
      name = "Date",
      guide = guide_colorbar(
        title.position = "top",
        barwidth = 10,
        barheight = 0.5
      )
    ) +
    labs(
      title = "Annual Pattern of Time Covariate",
      # subtitle = paste(
      #   "Solstices and Equinoxes Highlighted\n",
      #   "Winter Solstice (Dec 21), Spring Equinox (Mar 20),",
      #   "Summer Solstice (Jun 21), Fall Equinox (Sep 22)"
      # ),
      x = "Time of Day (24-hour)",
      y = "Covariate Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(hjust = 0.5),
      legend.text = element_text(angle = 45, hjust = 1)
    )

  
  # Print the plot
  print(p)
  
  # Return the data for further analysis if needed
  return(all_results)
}

# Run the test
results <- test_annual_covariate()