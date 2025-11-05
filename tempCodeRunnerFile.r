# ==============================================================================
# Multi-Country Energy Demand Forecasting (Automated Workflow)
# ==============================================================================
# This script simulates energy data for three distinct climate types,
# automates the forecasting process using the Prophet package, and
# visualizations all forecasts on a single, clean multi-panel chart.
#
# Libraries required:
# - prophet: For robust time series forecasting with seasonality and holidays.
# - tidyverse: For data manipulation (dplyr) and plotting (ggplot2).
#
# To install the required packages, run:
# install.packages(c("prophet", "tidyverse"))
# ==============================================================================

# 1. Load Required Libraries
library(tidyverse)
library(prophet)

# Set seed for reproducibility
set.seed(42)

# ==============================================================================
# 2. Automated Multi-Country Data Simulation (Mimicking API Data Structure)
# ==============================================================================
# This section simulates 3 years of daily data for three countries (Cold, Moderate, Hot)
# to reflect different climate-driven energy consumption patterns.

simulate_country_data <- function(country_name, start_date, years, base_level, temp_sensitivity, seasonality_shift) {
  # Create a daily time index
  time_index <- seq(as.Date(start_date), by = "day", length.out = 365 * years)
  n <- length(time_index)

  # Days since start (for trend)
  days_since_start <- 1:n

  # --- Temperature Simulation ---
  # Base annual temperature cycle (sin function)
  annual_cycle <- sin(2 * pi * days_since_start / 365)
  temp_base <- 15 + 10 * annual_cycle
  temperature <- temp_base + rnorm(n, 0, 1.5)

  # Adjust temperature based on country type
  if (country_name == "Hot Country") {
    temperature <- temperature + 5 # Generally hotter climate
  } else if (country_name == "Cold Country") {
    temperature <- temperature - 5 # Generally colder climate
  }

  # --- Energy Consumption Simulation ---
  # Consumption is driven by a mix of factors:
  # 1. Base Level + Trend (0.05 per day)
  # 2. Seasonality (annual sine wave)
  # 3. Temperature (U-shaped relationship: high demand in extreme cold/hot)
  # 4. Noise

  # Calculate Heating Degree Days (HDD) - demand when T is below a threshold (e.g., 18C)
  hdd <- pmax(0, 18 - temperature)
  # Calculate Cooling Degree Days (CDD) - demand when T is above a threshold (e.g., 24C)
  cdd <- pmax(0, temperature - 24)

  # Consumption model
  consumption <- base_level +
    0.05 * days_since_start + # Trend
    10 * annual_cycle +       # Strong baseline seasonality
    temp_sensitivity * (hdd + cdd) + # Temp-driven demand
    rnorm(n, 0, 3) # Noise

  # Combine into the required Prophet/time series structure
  data.frame(
    country = country_name,
    ds = time_index, # Date for Prophet
    y = consumption, # Target variable for Prophet
    temp = temperature # External regressor
  )
}

# Generate data for three countries
start_date <- "2021-01-01"
years_of_data <- 3

data_list <- list(
  simulate_country_data("Cold Country", start_date, years_of_data, 150, 5, 0),
  simulate_country_data("Moderate Country", start_date, years_of_data, 120, 3, 0),
  simulate_country_data("Hot Country", start_date, years_of_data, 100, 7, 0)
)

# Combine all data frames into one (Tidy Data)
full_data <- bind_rows(data_list)

# ==============================================================================
# 3. Forecasting Automation Loop (Using Prophet)
# ==============================================================================

# Define the forecast horizon
H <- 30 # Forecast 30 days into the future

# Find the last observed date for future forecasting
last_date <- max(full_data$ds)
future_dates <- seq(last_date + 1, last_date + H, by = "day")

# Function to fit Prophet model and generate forecast for a single country
forecast_single_country <- function(df_country) {
  # A. Setup the model and add temperature as an external regressor
  model <- prophet(
    df_country,
    daily.seasonality = TRUE,
    weekly.seasonality = TRUE,
    yearly.seasonality = TRUE
  )
  # Add the 'temp' regressor (Crucial for a demand model)
  model <- add_regressor(model, 'temp')

  # B. Prepare future data frame
  future_df <- data.frame(
    ds = future_dates
  )

  # C. Simulate (or load from a weather API) future temperature data (Crucial for regressors)
  # For demonstration, we'll use the average temperature trend from the last 30 days
  # and add noise. In a real app, this would be a weather forecast API call.
  temp_past_30_days <- df_country %>%
    filter(ds >= max(ds) - 30) %>%
    pull(temp)

  # Create a simple future temperature forecast (e.g., mean of recent history)
  future_df$temp <- mean(temp_past_30_days) + rnorm(H, 0, 1.5)


  # D. Predict and combine results
  forecast_results <- predict(model, future_df) %>%
    # Select only the relevant columns for plotting
    select(ds, yhat, yhat_lower, yhat_upper) %>%
    # Add a 'type' column to distinguish it from historical data
    mutate(type = "Forecast")

  return(forecast_results)
}

# Use the 'nest-map-unnest' pattern from tidyverse to automate the process
# 1. Group the full data by country and store the dataframes in a list column ('data')
nested_data <- full_data %>%
  group_by(country) %>%
  nest()

# 2. Apply the forecast function to each nested dataframe
# (This is the 'automated' part that replaces manual fitting)
forecasts_all <- nested_data %>%
  mutate(forecast = map(data, forecast_single_country)) %>%
  # 3. Unnest the forecast results back into a single tidy dataframe
  unnest(forecast)

# ==============================================================================
# 4. Prepare for Visualization and Plotting
# ==============================================================================

# Combine historical data and forecasts for easy plotting
historical_plot_data <- full_data %>%
  # Filter to show only the last 6 months of history + the full forecast range
  filter(ds >= max(ds) - 180) %>%
  rename(yhat = consumption) %>%
  mutate(
    type = "Historical",
    # Set confidence intervals to NA for historical points
    yhat_lower = NA_real_,
    yhat_upper = NA_real_
  ) %>%
  # Select only the required columns to match the structure of forecasts_all
  select(country, ds, yhat, yhat_lower, yhat_upper, type)

# Final combined data for plotting
plot_data <- bind_rows(historical_plot_data, forecasts_all)

# ==============================================================================
# 5. Multi-Country Visualization (ggplot2)
# ==============================================================================

energy_plot <- ggplot(plot_data, aes(x = ds, y = yhat, color = type)) +
  # Confidence Interval (only visible for the 'Forecast' type)
  geom_ribbon(
    aes(ymin = yhat_lower, ymax = yhat_upper, fill = type),
    alpha = 0.2,
    color = NA
  ) +
  # Plot the actual/forecast line
  geom_line(linewidth = 0.8) +
  # Separate plots for each country (Automated multi-graph)
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  # Style and labels
  labs(
    title = "Automated Daily Energy Demand Forecast by Country",
    subtitle = paste0("Prophet Model | Forecast Horizon: ", H, " days"),
    x = "Date",
    y = "Energy Consumption (MWh)",
    color = "Data Type",
    fill = "95% Confidence"
  ) +
  scale_color_manual(values = c("Historical" = "#3182CE", "Forecast" = "#D95F02")) +
  scale_fill_manual(values = c("Historical" = "#3182CE", "Forecast" = "#D95F02")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 16, color = "#2D3748"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

# Print the plot
print(energy_plot)

# ==============================================================================
# Optional: Print numerical summary of the last 5 forecast days for each country
# ==============================================================================

cat("\n--- Numerical Forecast Summary (Last 5 Days) ---\n")
forecasts_all %>%
  group_by(country) %>%
  slice_tail(n = 5) %>%
  mutate(
    ds = as.character(ds),
    yhat = round(yhat, 2),
    yhat_lower = round(yhat_lower, 2),
    yhat_upper = round(yhat_upper, 2)
  ) %>%
  print(n = 15) # Print all 5 rows for 3 countries
