
# Import & clean
rm(list = ls())
setwd("Downloads/projects")
hr <- read.csv("emp_attrition.csv")
hr <- hr[, !(names(hr) %in% c("Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"))]
cat_cols <- c("BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
hr[cat_cols] <- lapply(hr[cat_cols], factor)
hr$Attrition <- ifelse(hr$Attrition == "Yes", TRUE, FALSE)

# Cross-validation functions
train_test_generate <- function (df, seed = 123, proportion = 0.7) {
  set.seed(seed)
  size <- round(proportion * nrow(df))
  idx <- sample(seq_len(nrow(df)), size = size, replace = FALSE)
  return(list(df[idx, ], df[-idx, ]))
}

rsf_model <- function (df) {
  library(randomForestSRC)
  survival_object <- with(df, Surv(YearsAtCompany, Attrition))
  model <- rfsrc(Surv(YearsAtCompany, Attrition) ~ ., data = df, ntree = 100)
  return(model)
}

rsf_top_risk_individual <- function (train, test) {
  # Take some people so that the true capture is 50%.
  # Top 20% seem to be enough
  rsfm <- rsf_model(train)
  hazard_score <- predict(rsfm, test, importance = TRUE)$predicted
  sorted_test <- test[order(hazard_score, decreasing = TRUE), ]
  proportion <- 0.2
  size <- round(proportion * nrow(test))
  top_risk_indv <- sorted_test[1:size, ]
  return(top_risk_indv)
}

rsf_top_important_var <- function (df) {
  # Take variables with distinctively higher importance.
  # By manual check, 5 seems to be enough.
  rsfm <- rsf_model(df)
  imp_score <- predict(rsfm, importance = TRUE)$importance
  sorted_imp_score <- imp_score[order(imp_score, decreasing = TRUE)]
  num <- 10
  top_var <- sorted_imp_score[1:num]
  return(top_var)
}

rsf_performance_score <- function(train, test) {
  # How accurate are we doing
  top_risk_indv <- rsf_top_risk_individual(train, test)
  true_capture_rate <- sum(top_risk_indv$Attrition == TRUE)/sum(test$Attrition == TRUE)
  false_avoidance_rate <- 1 - sum(top_risk_indv$Attrition == FALSE)/sum(test$Attrition == FALSE)
  return(c(true_capture_rate, false_avoidance_rate))
}

cross_val <- function (k = 10) {
  true_capture_rate <- numeric(k)
  false_avoidance_rate <- numeric(k)
  # rank variables
  for (i in 1:k) {
    tg <- train_test_generate(hr, i)
    train_hr <- tg[[1]]
    test_hr <- tg[[2]]
    pfm_score <- rsf_performance_score(train_hr, test_hr)
    true_capture_rate[i] <- pfm_score[1]
    false_avoidance_rate[i] <- pfm_score[2]
    
  }
  sprintf("Our random survival forest model averagely have %.2f true capture rate (sd: %.2f) \
          and %.2f false capture rate (sd: %.2f)",
          mean(true_capture_rate), sd(true_capture_rate),
          mean(false_avoidance_rate), sd(false_avoidance_rate))
}

# General process
cross_val()
tg <- train_test_generate(hr)
train_hr <- tg[[1]]
test_hr <- tg[[2]]
rsf_top_important_var(train_hr)

# Visual: soon
library(ggplot2)
# Next: predicting how long are people going to stay
