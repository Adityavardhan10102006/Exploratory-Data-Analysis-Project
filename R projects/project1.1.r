# ==============================================================================
# Multi-Country Energy Demand Forecasting (Automated Workflow)
# ==============================================================================

# 1. Load Required Libraries
library(tidyverse)
library(prophet)
library(tidyr)
library(ggplot2)

set.seed(42)

# ==============================================================================
# 2. Simulate Multi-Country Energy Data
# ==============================================================================

simulate_country_data <- function(country_name, start_date, years, base_level, temp_sensitivity, seasonality_shift) {
  time_index <- seq(as.Date(start_date), by = "day", length.out = 365 * years)
  n <- length(time_index)
  days_since_start <- 1:n

  annual_cycle <- sin(2 * pi * days_since_start / 365)
  temp_base <- 15 + 10 * annual_cycle
  temperature <- temp_base + rnorm(n, 0, 1.5)

  if (country_name == "Hot Country") {
    temperature <- temperature + 5
  } else if (country_name == "Cold Country") {
    temperature <- temperature - 5
  }

  hdd <- pmax(0, 18 - temperature)
  cdd <- pmax(0, temperature - 24)

  consumption <- base_level +
    0.05 * days_since_start +
    10 * annual_cycle +
    temp_sensitivity * (hdd + cdd) +
    rnorm(n, 0, 3)

  data.frame(
    country = country_name,
    ds = time_index,
    y = consumption,
    temp = temperature
  )
}

start_date <- "2021-01-01"
years_of_data <- 3

data_list <- list(
  simulate_country_data("Cold Country", start_date, years_of_data, 150, 5, 0),
  simulate_country_data("Moderate Country", start_date, years_of_data, 120, 3, 0),
  simulate_country_data("Hot Country", start_date, years_of_data, 100, 7, 0)
)

full_data <- bind_rows(data_list)

# ==============================================================================
# 3. Forecasting Automation with Prophet
# ==============================================================================

H <- 30
last_date <- max(full_data$ds)
future_dates <- seq(last_date + 1, last_date + H, by = "day")

forecast_single_country <- function(df_country) {
  model <- prophet(
    daily.seasonality = TRUE,
    weekly.seasonality = TRUE,
    yearly.seasonality = TRUE
  )
  model <- add_regressor(model, 'temp')
  model <- fit.prophet(model, df_country)

  future_df <- data.frame(ds = future_dates)

  temp_past_30_days <- df_country %>%
    filter(ds >= max(ds) - 30) %>%
    pull(temp)

  future_df$temp <- mean(temp_past_30_days) + rnorm(H, 0, 1.5)

  forecast_results <- predict(model, future_df) %>%
    select(ds, yhat, yhat_lower, yhat_upper) %>%
    mutate(type = "Forecast")

  return(forecast_results)
}

nested_data <- full_data %>%
  group_by(country) %>%
  nest()

forecasts_all <- nested_data %>%
  mutate(forecast = map(data, forecast_single_country)) %>%
  unnest(forecast)

# ==============================================================================
# 4. Prepare Data for Visualization
# ==============================================================================

historical_plot_data <- full_data %>%
  filter(ds >= max(ds) - 180) %>%
  rename(yhat = y) %>%
  mutate(
    type = "Historical",
    yhat_lower = NA_real_,
    yhat_upper = NA_real_
  ) %>%
  select(country, ds, yhat, yhat_lower, yhat_upper, type)

plot_data <- bind_rows(historical_plot_data, forecasts_all)

# ==============================================================================
# 5. Visualization (Display + Save for VS Code)
# ==============================================================================

energy_plot <- ggplot(plot_data, aes(x = ds, y = yhat, color = type)) +
  geom_ribbon(
    aes(ymin = yhat_lower, ymax = yhat_upper, fill = type),
    alpha = 0.2,
    color = NA
  ) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
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

# ✅ Display the plot in VS Code
print(energy_plot)

# ✅ Save the plot as PNG
ggsave("energy_forecast.png", plot = energy_plot, width = 10, height = 8, dpi = 300)

# ==============================================================================
# 6. Numerical Forecast Summary
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
  print(n = 15)
