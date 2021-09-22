library('ggpubr')
library('tidyverse')



slopes <- read.csv('~/OneDrive/Data/FAN_ET/Badanie_P/2017-05-06_Badanie_P/BadanieP_FAN_ET/Scripts/fan_slopes.csv')
WMC <- read.csv('~/OneDrive/Data/FAN_ET/Badanie_P/2017-05-06_Badanie_P/BadanieP_FAN_ET/results/ID_GF_WMC.csv')
merged = tibble(merge(slopes, WMC, by="PART_ID"))

merged <- filter(merged, LEV_EASY_CORR_Slope != 0,  LEV_MED_CORR_Slope != 0, LEV_HARD_CORR_Slope != 0)

shapiro.test(merged$LEV_EASY_CORR_Slope)
shapiro.test(merged$LEV_MED_CORR_Slope)
shapiro.test(merged$LEV_HARD_CORR_Slope)
shapiro.test(merged$WMC)

ggqqplot(merged$LEV_EASY_CORR_Slope, ylab = "Slopes for Low RM trials")
ggqqplot(merged$LEV_MED_CORR_Slope, ylab = "Slopes for Med RM trials")
ggqqplot(merged$LEV_HARD_CORR_Slope, ylab = "Slopes for High RM trials")
ggqqplot(merged$WMC, ylab = "WMC")


ggscatter(merged, x = "LEV_EASY_CORR_Slope", y = "WMC", add = "reg.line",
          conf.int = TRUE, cor.coef = TRUE, cor.method = "spearman",
          xlab = "Slopes for Low RM trials", ylab = "Working Memeory Capacity")
ggsave("low_slopes.png", dpi=300)

ggscatter(merged, x = "LEV_MED_CORR_Slope", y = "WMC", add = "reg.line",
          conf.int = TRUE, cor.coef = TRUE, cor.method = "spearman",
          xlab = "Slopes for Medium RM trials", ylab = "Working Memeory Capacity")
ggsave("med_slopes.png", dpi=300)

ggscatter(merged, x = "LEV_HARD_CORR_Slope", y = "WMC", add = "reg.line",
          conf.int = TRUE, cor.coef = TRUE, cor.method = "spearman",
          xlab = "Slopes for High RM trials", ylab = "Working Memeory Capacity")
ggsave("hard_slopes.png", dpi=300)
