library(dplyr)
library(ggplot2)
library(patchwork)

data <- read.csv("/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order_w_depth.csv")
summary <- read.csv("/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order_summary_stat.csv")

# Minimal theme setup
minimal_theme <- theme_minimal(base_size = 8) + 
  theme(
    panel.background = element_rect(fill="white", color=NA),
    panel.grid = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_text(size=6),
    axis.text.x = element_text(size=6)
  )

p1 <- data %>%
  ggplot(aes(NND)) +
  geom_histogram(bins = 6, fill="skyblue", color=NA) +
  minimal_theme +
  labs(x = "Nearest Neighbor Distance (in body lengths)")

p2 <- data %>%
  ggplot(aes(dist_from_centre)) +
  geom_histogram(bins = 6, fill="lightgreen", color=NA) +
  minimal_theme +
  labs(x = "Distance from school centroid (in body lengths)",
       title = paste0("School cohesion = ", round(summary$group_cohesion,3)))

p3 <- data %>%
  ggplot(aes(heading_nn)) +
  geom_histogram(bins = 6, fill="salmon", color=NA) +
  minimal_theme +
  labs(x = "Alignment to nearest neighbor",
       title = paste0())

p4 <- data %>%
  ggplot(aes(heading_rel_to_group)) +
  geom_histogram(bins = 6, fill="orange", color=NA) +
  minimal_theme +
  labs(x = "Individual alignment to school heading",
       title = paste0("School polarization/alignment = ", round(summary$polarisation,3)))

# Combine plots horizontally
combined_plot <- (p1 | p2) / (p3 | p4)
combined_plot
