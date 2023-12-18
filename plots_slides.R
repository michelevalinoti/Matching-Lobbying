# Set a seed for reproducibility
set.seed(42)

# Generate a sample of 1000 points from a standard normal distribution
data <- rnorm(1000)

# Define a common x-axis range for all plots
x_range <- range(data)

# Function to add vertical lines to the histogram
add_vertical_lines <- function(data, hist_obj, col = "blue", lwd = 1) {
  for (i in 1:100) {
    x_value <- data[i]
    if (!is.na(x_value)) {  # Check for missing values
      in_range <- x_value >= min(hist_obj$mids) & x_value <= max(hist_obj$mids)
      if (in_range) {
        lines(c(x_value, x_value), c(0, dnorm(x_value, mean = mean(data), sd = sd(data))), col = col, lwd = lwd)
      }
    }
  }
}

# Function to create and save a plot with a common x-axis range
save_plot <- function(data, title, filename, x_range) {
  # Set up larger plot size
  options(
    repr.plot.width = 8,  # Adjust the width as needed
    repr.plot.height = 6  # Adjust the height as needed
  )
  
  # Create the plot with the common x-axis range
  hist_obj <- hist(data, breaks = 30, plot = FALSE)
  hist(data, breaks = 30, prob = TRUE, col = "lightblue", xlab = "Relevance", main = title, xlim = x_range, cex = 1.5, mar = c(5, 5, 4, 2))
  curve(dnorm(x, mean = mean(data), sd = sd(data)), col = "red", lwd = 2, add = TRUE)
  add_vertical_lines(data, hist_obj)
  segments(x0 = 1, y0 = 0, x1 = 1, y1 = 0.4, col = "black", lwd = 4)
  mtext(expression(paste(underline(r[i])), sep = ""), side = 1, line = 0, at = 1.2, cex = 1)
  
  # Open a PNG graphics device
  png(filename, width = 800, height = 600, units = "px", res = 300)  # Adjust the width, height, and dpi as needed
  # Print the plot to the device
  dev.off()
}

# Save three plots with the common x-axis range

# 1. Randomly drawn points
save_plot(data, "Randomly Drawn Points", "random_points.png", x_range)

# 2. Points with Mean 1
data_mean1 <- rnorm(100, mean = 1)
save_plot(data_mean1, "Bills based lawmakers' information set", "points_mean1.png", x_range)

# 3. Truncated at -2
data_truncated <- rnorm(100, mean=1.5)
#data_truncated <- data_truncated[data_truncated >= 0]  # Apply truncation
save_plot(data_truncated, "Bills based lawmakers' information set and lobbyists", "truncated_points.png", x_range)

# 4. 2000 points with 1000 randomly selected
data_double <- rnorm(2000)
random_indices <- sample(1:2000, 1000)
data_double <- data_double[random_indices]
save_plot(data_double, "Random draws from 2000 points", "double_sample.png", x_range)

# 5. 2000 points with mean 0.5 and 100 randomly selected
data_double_mean05 <- rnorm(2000, mean = 0.5)
random_indices_mean05 <- sample(1:2000, 100)
data_double_mean05 <- data_double_mean05[random_indices_mean05]
save_plot(data_double_mean05, "Random draws 2000 points with mean 0.5", "double_sample_mean05.png", x_range)
