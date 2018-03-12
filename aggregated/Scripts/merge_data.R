library(readxl)
library(Hmisc)
metrics <- read.csv("../results/metrics.csv")
# WMC <- read.csv('../results/WMC.csv', sep=';')
aggr_beh <- read_excel("../results/aggr_beh.xlsx", col_types = c("numeric", "blank", "blank", "text", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "blank", "numeric", "numeric", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "numeric", "text", "numeric", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "numeric", "numeric", "blank", "numeric", 
                                                      "numeric", "numeric", "blank", "numeric", 
                                                      "numeric", "numeric", "blank", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "numeric", "numeric", "numeric", "numeric", 
                                                      "blank", "blank", "blank", "numeric", 
                                                       "numeric", "numeric", "numeric"))

data <- merge(metrics, aggr_beh, by.x="PART_ID", by.y="KOD")
# data <- merge(data, WMC, by.x = "PART_ID", by.y="Part_id")
data$mc <- data$MC_4 - (1 - data$MC_123) / 3
data$csp <- (data$CS_LET_4 + data$CS_LET_6 + data$CS_LET_8) / 3
data$arr <- (data$ARR_LET_5 + data$ARR_LET_7 + data$ARR_LET_9) / 3

# data$EASY_DUR_1 <- ((data$RV_AVG_DUR_EASY_CORR_3 + data$RV_AVG_DUR_EASY_CORR_4) / 2) / ((data$RV_AVG_DUR_EASY_CONTROL_3 + data$RV_AVG_DUR_EASY_CONTROL_4) / 2)
#data$EASY_DUR_2 <- ((data$RV_AVG_DUR_EASY_CORR_3 + data$RV_AVG_DUR_EASY_CORR_4) / 2) / ((data$RV_AVG_DUR_EASY_CONTROL_3 + data$RV_AVG_DUR_EASY_CONTROL_4) / 2)
# data$MED_DUR_1 <- ((data$RV_AVG_DUR_MEDIUM_CORR_3 + data$RV_AVG_DUR_MEDIUM_CORR_4) / 2) / ((data$RV_AVG_DUR_MEDIUM_CONTROL_3 + data$RV_AVG_DUR_MEDIUM_CONTROL_4) / 2)
#data$MED_DUR_2 <- (((data$RV_AVG_DUR_MEDIUM_CORR_3 + data$RV_AVG_DUR_MEDIUM_CORR_4) / 2 + (data$RV_AVG_DUR_MEDIUM_SMALL_ERROR_3 + data$RV_AVG_DUR_MEDIUM_SMALL_ERROR_4) / 2) / 2) / ((data$RV_AVG_DUR_MEDIUM_CONTROL_3 + data$RV_AVG_DUR_MEDIUM_CONTROL_4) / 2)
#data$HARD_DUR_1 <- ((data$RV_AVG_DUR_HARD_CORR_3 + data$RV_AVG_DUR_HARD_CORR_4) / 2) / ((data$RV_AVG_DUR_HARD_CONTROL_3 + data$RV_AVG_DUR_HARD_CONTROL_4) / 2)
#data$HARD_DUR_2 <- (data$RV_AVG_DUR_HARD_BIG_ERROR_3 + data$RV_AVG_DUR_HARD_BIG_ERROR_4 + data$RV_AVG_DUR_HARD_CONTROL_3 + data$RV_AVG_DUR_HARD_CONTROL_4 + data$RV_AVG_DUR_HARD_CORR_3 + data$RV_AVG_DUR_HARD_CORR_4 + data$RV_AVG_DUR_HARD_SMALL_ERROR_3 + data$RV_AVG_DUR_HARD_SMALL_ERROR_4) / (data$RV_AVG_DUR_HARD_CONTROL_3 + data$RV_AVG_DUR_HARD_CONTROL_4)
#data$EASY_FIX_1 <- (data$RV_SUM_FIX_EASY_CORR_3 + data$RV_SUM_FIX_EASY_CORR_4) / (data$RV_SUM_FIX_EASY_BIG_ERROR_3 + data$RV_SUM_FIX_EASY_BIG_ERROR_4 + data$RV_SUM_FIX_EASY_CONTROL_3 + data$RV_SUM_FIX_EASY_CONTROL_4 + data$RV_SUM_FIX_EASY_CORR_3 + data$RV_SUM_FIX_EASY_CORR_4 + data$RV_SUM_FIX_EASY_SMALL_ERROR_3 + data$RV_SUM_FIX_EASY_SMALL_ERROR_4)
#data$EASY_FIX_2 <- (data$RV_SUM_FIX_EASY_CORR_3 + data$RV_SUM_FIX_EASY_CORR_4) / (data$RV_SUM_FIX_EASY_BIG_ERROR_3 + data$RV_SUM_FIX_EASY_BIG_ERROR_4 + data$RV_SUM_FIX_EASY_CONTROL_3 + data$RV_SUM_FIX_EASY_CONTROL_4 + data$RV_SUM_FIX_EASY_CORR_3 + data$RV_SUM_FIX_EASY_CORR_4 + data$RV_SUM_FIX_EASY_SMALL_ERROR_3 + data$RV_SUM_FIX_EASY_SMALL_ERROR_4)
#data$MED_FIX_1 <- (data$RV_SUM_FIX_MEDIUM_CORR_3 + data$RV_SUM_FIX_MEDIUM_CORR_4) / (data$RV_SUM_FIX_MEDIUM_BIG_ERROR_3 + data$RV_SUM_FIX_MEDIUM_BIG_ERROR_4 + data$RV_SUM_FIX_MEDIUM_CONTROL_3 + data$RV_SUM_FIX_MEDIUM_CONTROL_4 + data$RV_SUM_FIX_MEDIUM_CORR_3 + data$RV_SUM_FIX_MEDIUM_CORR_4 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_3 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_4)
#data$MED_FIX_2 <- (data$RV_SUM_FIX_MEDIUM_CORR_3 + data$RV_SUM_FIX_MEDIUM_CORR_4 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_3 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_4) / (data$RV_SUM_FIX_MEDIUM_BIG_ERROR_3 + data$RV_SUM_FIX_MEDIUM_BIG_ERROR_4 + data$RV_SUM_FIX_MEDIUM_CONTROL_3 + data$RV_SUM_FIX_MEDIUM_CONTROL_4 + data$RV_SUM_FIX_MEDIUM_CORR_3 + data$RV_SUM_FIX_MEDIUM_CORR_4 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_3 + data$RV_SUM_FIX_MEDIUM_SMALL_ERROR_4)
# data$HARD_FIX_1 <- (data$RV_SUM_FIX_HARD_CORR_3 + data$RV_SUM_FIX_HARD_CORR_4) / (data$RV_SUM_FIX_HARD_BIG_ERROR_3 + data$RV_SUM_FIX_HARD_BIG_ERROR_4 + data$RV_SUM_FIX_HARD_CONTROL_3 + data$RV_SUM_FIX_HARD_CONTROL_4 + data$RV_SUM_FIX_HARD_CORR_3 + data$RV_SUM_FIX_HARD_CORR_4 + data$RV_SUM_FIX_HARD_SMALL_ERROR_3 + data$RV_SUM_FIX_HARD_SMALL_ERROR_4)
#data$HARD_FIX_2 <- (data$RV_SUM_FIX_HARD_BIG_ERROR_3 + data$RV_SUM_FIX_HARD_BIG_ERROR_4+data$RV_SUM_FIX_HARD_CORR_3 + data$RV_SUM_FIX_HARD_CORR_4 + data$RV_SUM_FIX_HARD_SMALL_ERROR_3 + data$RV_SUM_FIX_HARD_SMALL_ERROR_4) / (data$RV_SUM_FIX_HARD_BIG_ERROR_3 + data$RV_SUM_FIX_HARD_BIG_ERROR_4 + data$RV_SUM_FIX_HARD_CONTROL_3 + data$RV_SUM_FIX_HARD_CONTROL_4 + data$RV_SUM_FIX_HARD_CORR_3 + data$RV_SUM_FIX_HARD_CORR_4 + data$RV_SUM_FIX_HARD_SMALL_ERROR_3 + data$RV_SUM_FIX_HARD_SMALL_ERROR_4)


#data$ALLOC_EASY <- (scale(data$EASY_DUR_2) + scale(data$EASY_FIX_2)) / 2 
#data$ALLOC_MED <- (scale(data$MED_DUR_2) + scale(data$MED_FIX_2)) / 2 
#data$ALLOC_HARD <- (scale(data$HARD_DUR_2) + scale(data$HARD_FIX_2)) / 2 

data$X <- NULL

gf_factors <- c('RAV', 'TAO', 'FIG')
gf <- data[gf_factors]
gf.fact <- factanal(x=gf, factors = 1, rotation = 'varimax', scores = 'regression')
data$GF <- gf.fact$scores

wmc <- data[c('mc', 'csp', 'arr')]
wmc.fact <- factanal(x=wmc, factors = 1, rotation = 'varimax', scores = 'regression')
data$WMC <- wmc.fact$scores

write.csv(data, file='FAN_ET_aggr.csv', sep=',')
#rcorr(as.matrix(data[c('NT_EASY', 'RTM_EASY')]), type="pearson")
#library("ggpubr")
#ggscatter(data, x = "NT_EASY", y = "RTM_EASY", 
#          add = "reg.line", conf.int = TRUE, 
#          cor.coef = TRUE, cor.method = "pearson",
#          xlab = "NT_EASY", ylab = "RTM_EASY")

# Pricipal Components Analysis
# entering raw data and extracting PCs 
# from the correlation matrix 

# 
fit <- princomp(data[gf_factors], cor=TRUE)
summary(fit) # print variance accounted for 
loadings(fit) # pc loadings 
plot(fit,type="lines") # scree plot
plot(fit)
#data$wmc <- fit$scores # the principal components
#biplot(fit)




