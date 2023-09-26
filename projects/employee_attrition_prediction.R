
# Import & clean
rm(list = ls())
hr <- read.csv("HR_Analytics.csv")
hr <- hr[, !(names(hr) %in% c("Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"))]
cat_cols <- c("BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
hr[cat_cols] <- lapply(hr[cat_cols], factor)
hr$Attrition <- ifelse(hr$Attrition == "Yes", TRUE, FALSE)

# Function to divide data
training_gen <- function (df, seed, proportion = 0.7) {
  set.seed(seed)
  size <- round(proportion * nrow(df))
  idx <- sample(seq_len(nrow(df)), size = size, replace = FALSE)
  return(list(df[idx, ], df[-idx, ]))
}

# Random forest
library(randomForest)
tg_hr <- training_gen(hr, 123)
train_hr <- tg_hr[[1]]
test_hr <- tg_hr[[2]]
rf_model <- randomForest(Attrition ~ ., train_hr, ntree = 100, importance = TRUE)
# Feature selection based on importance
imp_score <- importance(rf_model)[, 2]
imp_threshold <- 4
train_hr_featured <- train_hr[, c("Attrition", names(which(imp_score >= imp_threshold)))]
rf_model_featured <- randomForest(Attrition ~ ., train_hr_featured, ntree = 100, importance = TRUE)
# I want the prediction to capture around 50% true attrition
#   I had to edit the threshold to 0.25
pred_prob_rf <- predict(rf_model_featured, test_hr, type = "response")
pred_attrition_rf <- ifelse(pred_prob_rf >= 0.25, TRUE, FALSE)
table(pred_attrition_rf, test_hr$Attrition)

# Survival analysis, using random forest output as input
library(survival)
hr$rf <- predict(rf_model_featured, hr)
# Get new training set
tg_hr <- training_gen(hr, 89)
train_hr <- tg_hr[[1]]
test_hr <- tg_hr[[2]]
# Prediction: much more accurate
survival_obj <- with(train_hr, Surv(YearsAtCompany, Attrition))
cox_model <- coxph(Surv(YearsAtCompany, Attrition) ~ ., data = train_hr)
pred_prob_cox <- predict(cox_model, test_hr, type = "expected")
pred_attrition_cox <- ifelse(pred_prob_cox >= 0.5, TRUE, FALSE)
table(pred_attrition_cox, test_hr$Attrition)

# Visualising:
library(ggplot2)
data <- data.frame(
  Variable = factor(c("Age", "Education", "Experience", "RandomForestOutput")),
  Importance = c(0.4, 0.3, 0.5, 0.6) # Example importance values
)
# Plot variable importance with a corporate-style theme
ggplot(data, aes(x = Variable, y = Importance, fill = Variable)) +
  geom_bar(stat = "identity") +
  theme_minimal() +  # Or use a custom theme to match your corporate style
  labs(title = "Variable Importance for Employee Attrition",
       x = "Variables",
       y = "Importance")

# Comparing factors in Cox
cox_summary <- summary(cox_model)
p_val <- cox_summary$coefficients[, "Pr(>|z|)"]
significant_var <- names(p_val[p_val < 0.05])
hazard_ratio <- cox_summary$coefficients[significant_var, "exp(coef)"]
hazard_ratio
# It seems that longer career track, satisfaction, non-marketing education are
#  associated with lower risk of quitting. While work travel and company change 
#  are associated with higher risk of quitting.
# Surprisingly, higher work life balance seems to increase attrition risk, but it can
#   be due to retirement age