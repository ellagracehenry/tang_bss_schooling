library(dplyr)
library(lme4)
library(glmmTMB)
library(ggmosaic)

datagroupAO <- read.csv("/Users/ellag/Downloads/bss_schooling_data.xlsx - group_metrics_new.csv")
datagroup <- read.csv("/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/data/finished_data_0309/bss_schooling_data/output/summary_global.csv")
dataAO <- read.csv("/Users/ellag/Downloads/bss_schooling_data.xlsx - ind_data_new.csv")
data_final <- read.csv("/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/data/finished_data_0309/bss_schooling_data/output/individual_global.csv")

datafinal1 <- merge(data_final, dataAO[, c("image_name", "individual_ID", "annotated_group_size", "infected", "spot_count")])
datagroup1 <- merge(datagroup, datagroupAO[, c("image_name","image_ID", "group_size", "number_infected", "spot_count")])

datagroup1$polarisation
datagroup1$percapspot <- datagroup1$spot_count/datagroup1$group_size
datagroup1$propinf <- datagroup1$number_infected/datagroup1$group_size

datagroup1$polarisation
datagroupAO$percapspot <- datagroupAO$spot_count/datagroupAO$group_size
datagroupAO$propinf <- datagroupAO$number_infected/datagroupAO$group_size

hist(datagroup1$percapspot)

#Polarisartion ~ per cap spot
datagroup1 %>%
  #filter(percapspot > 0) %>%
  ggplot(aes(x = percapspot, y = polarisation)) +
  geom_point(alpha = 0.7) +
  labs(x = "Average Spot Count", y = "Polarization") +
  theme_classic(base_size = 20) #+
  #geom_smooth()

#Polarisartion ~ prop_inf
datagroup1 %>%
  #filter(percapspot > 0) %>%
  ggplot(aes(x = propinf, y = polarisation)) +
  geom_point(alpha = 0.7) +
  labs(x = "Proportion of School Infected", y = "Polarization") +
  theme_classic(base_size = 20)
  #geom_smooth()

#Cohesion ~ per cap spot
datagroup1 %>%
  #filter(percapspot > 0) %>%
  ggplot(aes(x = percapspot, y = group_cohesion)) +
  geom_point(alpha = 0.7) +
  labs(x = "Average Spot Count", y = "Cohesion") +
  theme_classic(base_size = 20) #+
  #geom_smooth()

#Polarisartion ~ prop_inf
datagroup1 %>%
  #filter(percapspot > 0) %>%
  ggplot(aes(x = propinf, y = group_cohesion)) +
  geom_point(alpha = 0.7) +
  labs(x = "Proportion of School Infected", y = "Cohesion") +
  theme_classic(base_size = 20) 


summary(glm(polarisation ~ percapspot + group_size + number_infected, data = datagroup, family = beta()))

model <- glmmTMB(polarisation ~ percapspot + group_size + number_infected, data = datagroup1, family = beta_family(link = "logit"))
summary(model)


## NND
datafinal1 %>%
  filter(!is.na(infected)) %>%
  ggplot(aes(as.factor(infected), NND)) +
  geom_boxplot(alpha = 0.3) +
  labs(x = "Infected") +
  theme_classic(base_size = 20) 

datafinal1 %>%
  filter(spot_count != 0) %>%
  ggplot(aes(spot_count, NND)) +
  geom_point(alpha = 0.3) +
  theme_classic(base_size = 20) 

summary(glm(NND ~ infected + annotated_group_size + (1|image_name), data = datafinal1))

## Distance from centre
datafinal1 %>%
  filter(!is.na(infected)) %>%
  ggplot(aes(as.factor(infected), dist_from_centre)) +
  geom_boxplot(alpha = 0.3) +
theme_classic(base_size = 20) +
  labs(x = "Infected", y = "Distance from center") 

datafinal1 %>%
  filter(!is.na(infected)) %>%
  ggplot(aes(spot_count, dist_from_centre)) +
  geom_point(alpha = 0.3) +
  theme_classic(base_size = 20) +
labs(x = "Spot Count", y = "Distance from center") 


##Distance to back
datafinal1 %>%
  filter(!is.na(infected)) %>%
  filter(norm_dist_to_back > 0) %>%
  ggplot(aes(as.factor(infected), norm_dist_to_back)) +
  geom_boxplot(alpha = 0.3) +
  theme_classic(base_size = 20) +
  labs(x = "Infected", y = "Distance from back") 

summary(lm(norm_dist_to_back ~ infected, data = datafinal1))

datafinal1 %>%
  filter(!is.na(infected)) %>%
  filter(norm_dist_to_back != 0) %>%
  ggplot(aes(spot_count, norm_dist_to_back)) +
  geom_point(alpha = 0.3) +
  theme_classic(base_size = 20) +
  labs(x = "Spot Count", y = "Distance from back") 

datafinal1 %>%
  filter(!is.na(infected)) %>%
  ggplot() +
  geom_mosaic(aes(x = product(infected), fill = back_ind)) +
  labs(x = "Spot Count", y = "Distance from back") +
  scale_fill_discrete() +
  theme_classic(base_size = 20) 

datafinal1 %>%
  filter(!is.na(infected)) %>%
  filter(spot_count != 0) %>%
  ggplot(aes(spot_count, back_ind)) +
  geom_point() +
  geom_smooth()

datagroup1 %>%
  



#Highest
datafinal1 <- datafinal1 %>%
  mutate(
    infected = factor(
      infected,
      levels = c(0, 1),
      labels = c("Not infected", "Infected")
    ),
    highest_ind = factor(
      highest_ind,
      levels = c(0, 1),
      labels = c("Not highest", "Highest")
    )
  )

datafinal1 %>%
  filter(!is.na(infected)) %>%
  ggplot() +
  geom_mosaic(aes(x = product(infected), fill = highest_ind)) +
  labs(x = "Spot Count", y = "Distance from back") +
  scale_fill_discrete() + 
  theme_classic(base_size = 20) 

table(datafinal1$infected, datafinal1$highest_ind)



datafinal1 %>%
  filter(!is.na(infected)) %>%
  filter(dist_to_highest > 0) %>%
  ggplot(aes(x = spot_count, y = norm_dist_to_highest)) +
  geom_point(alpha = 0.3)  + 
  theme_classic(base_size = 20) +
  labs(x = "Spot Count", y = "Distance to highest")


datafinal1 %>%
  filter(!is.na(infected)) %>%
  filter(norm_dist_to_highest > 0) %>%
  ggplot(aes(as.factor(infected), norm_dist_to_highest)) +
  geom_boxplot(alpha = 0.3) +
  theme_classic(base_size = 20) +
  labs(x = "Infected", y = "Distance from highest") 
