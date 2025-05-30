library(ggplot2)
library(lubridate)
library(dplyr)
library(suncalc)

# predictions <- read.csv('C:/Users/myset/Documents/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/RB_19_20250302_192735/predictions.csv')
predictions <- read.csv('~/1.Education/CowStudyApp/data/analysis_results/hmm/PRODUCT/Preds_Comb_19_DayNight_20250305_224041/predictions.csv')
predictions$is_day <- FALSE

timezone <- 'America/Denver'

local_times <- as.POSIXct(predictions$posix_time, origin="1970-01-01", tz=timezone)

unique_dates <- unique(as.Date(local_times))

unique_ID <- unique(predictions$ID)

buffer = 1.5 # hours

# Calculate sun times for each unique date

for (id in head(unique_ID,1)){ 
  for(date in head(unique_dates,1)) {
    date <- as.Date(date)
   
    mask <- ((as.Date(local_times) == date) & (predictions$ID == id))
    
    filtered_data <- predictions[mask,]
    
    day_data <- data.frame(
      date = date,
      lat = filtered_data$latitude[1],
      lon = filtered_data$longitude[1]
    )
     
    
    sun_times <- getSunlightTimes(
      data = day_data, 
      keep = c("sunrise", "sunset"), 
      tz = timezone
    )
    
    str(sun_times)

    
    print((sun_times$sunrise - (60*60)*buffer))
    predictions$is_day[mask] <- 
      local_times[mask] >= (sun_times$sunrise - (60*60)*buffer) & 
      local_times[mask] <= (sun_times$sunset + (60*60)*buffer)
    
  }
}

c <- sample(1:nrow(predictions), 10)
# Print those rows (all columns)
print(predictions[c,])


# Convert posix_time to Date and Time
local_times <- as.POSIXct(predictions$posix_time, origin="1970-01-01", tz=timezone)
predictions$date <- as.Date(local_times)
predictions$time <- format(local_times, "%H:%M:%S")

# Use dplyr to group and find first NIGHTTIME occurrence
library(dplyr)


night_starts <- predictions %>%
  filter(activity == "NIGHTTIME") %>%
  group_by(date) %>%
  summarize(
    first_night_posix = min(posix_time),
    first_night_tz = min(time)
  )


print(night_starts, n=5)



# Check sunset times for these dates
test_dates <- as.Date(c("2018-11-17", "2018-11-18", "2018-11-19", "2018-11-20", "2018-11-21"))

for(date in test_dates) {
  day_data <- data.frame(
    date = as.Date(date),
    lat = 45.59843,
    lon = -111.6300
  )
  
  sun_times <- getSunlightTimes(
    data = day_data, 
    keep = c("sunrise", "sunset"), 
    tz = timezone
  )
  
  cat("Date:", format(date), "\n")
  cat("Sunrise:", format(sun_times$sunrise, "%H:%M:%S"), "\n")
  cat("Sunset:", format(sun_times$sunset, "%H:%M:%S"), "\n")
  cat("------------------------\n")
}

