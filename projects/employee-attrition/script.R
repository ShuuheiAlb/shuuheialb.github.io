
# Import & clean
setwd("Downloads/shuuheialb.github.io/projects/employee-attrition")
rm(list = ls())
hr <- read.csv("hr_data.csv")

cols <- names(hr)
cols <- cols[cols != "EmployeeNumber"]
single_value_cols <- names(hr)[sapply(hr, function (col) length(unique(col)) == 1)]
cols <- cols[!(cols %in% single_value_cols)]

# Convert column data types
cat_cols <- c("BusinessTravel", "Department", "EducationField", "JobRole")
hr[cat_cols] <- lapply(hr[cat_cols], factor)
bool_cols <- c("Attrition", "Gender", "MaritalStatus", "OverTime")
hr[bool_cols] <- lapply(hr[bool_cols], function (col) ifelse(col == "Yes", TRUE, FALSE))


library(randomForestSRC)
library(survival)
library(pec, warn.conflicts = FALSE)
library(pROC)
library(ggplot2)
library(reshape2)
library(viridis)

# Base model
rsf_model <- function (df) {
  model <- rfsrc(Surv(YearsAtCompany, Attrition) ~ ., data = df, ntree = 100)
  return(model)
}

# Accuracy plot
plot_pec <- function(df, model) {
  pred_error <- pec(model, data = df, formula = Surv(YearsAtCompany, Attrition) ~ .,
                    splitMethod = "cv10", cens.model = "marginal")
  plot(pred_error, xlim = c(0, 20), )
  title("RSF Model Prediction Error Curve")
}

# Feature selection functions
# 1. Importance function
rsf_importance <- function (df, model, sort = TRUE) {
  importance <- predict(model, df, importance = TRUE)$importance
  if (sort) {
    importance <- importance[order(importance, decreasing = TRUE)]
  }
  return(importance)
}

# 2. Averaged importance rank
train_test_generate <- function (df, proportion = 0.7) {
  size <- round(proportion * nrow(df))
  idx <- sample(seq_len(nrow(df)), size = size, replace = FALSE)
  return(list("train" = df[idx, ], "test" = df[-idx, ]))
}
k_fold_cross_val <- function (df, k = 10) {
  for (fold in 1:k) {
    separate <- train_test_generate(df)
    train <- separate$train
    test <- separate$test
    model <- rsf_model(train)
    if (!(exists("importance_tot"))) {
      importance_tot <- rsf_importance(test, model, FALSE)
    } else {
      importance_tot <- importance_tot + rsf_importance(test, model, FALSE)
    }
  }
  print("Average variable importance:")
  print(importance_tot[order(importance_tot, decreasing = TRUE)]/length(importance_tot))
}

# 3. Correlation
plot_correlation <- function (df, model) {
  cor_matrix <- cor(df[, !(names(df) %in% cat_cols)])
  melted_cor_matrix <- melt(cor_matrix)
  melted_cor_matrix$value <- abs(melted_cor_matrix$value)
  ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() + scale_fill_viridis() +
    xlab("") + ylab("") + ggtitle("Correlation Map") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
          plot.title = element_text(hjust = 0.5))
}

# First iteration
#hr1 <- hr[cols]
#model_v1 <- rsf_model(hr1)
#plot_pec(hr1, model_v1)
#k_fold_cross_val(hr1)
#focus_cols <- c("Attrition", "YearsAtCompany", "OverTime", "JobLevel", "Age",
#                "NumCompaniesWorked", "MonthlyIncome", "StockOptionLevel")
#plot_correlation(hr[focus_cols], model_v1)

# Second iteration
hr["NumCompaniesWorkedPerYear"] <- hr["NumCompaniesWorked"]/(hr["TotalWorkingYears"] + 0.5)
cols <- c("Attrition", "YearsAtCompany", "MonthlyIncome", "OverTime", "Age",
          "JobRole", "NumCompaniesWorkedPerYear", "StockOptionLevel")
hr2 <- hr[cols]
k_fold_cross_val(hr2)
model_v2 <- rsf_model(hr2)
plot_pec(hr2, model_v2)

# Prediction
rsf_attrition_probability <- function (df, model) {
  return(predict(model, df)$chf)
}
rsf_high_risk_indiv_idx <- function (df, model, limit = 100) {
  df$HazardScore <- predict(model, df)$predicted
  return(order(df$HazardScore, decreasing = TRUE)[1:limit])
}

for (i in 1:5) {
  hr[, paste0("AttritionProb_Year", i)] <- rsf_attrition_probability(hr, model_v2)[, i+1]
}
head(hr)

hriidx <- rsf_high_risk_indiv_idx(hr, model_v2)
hridv <- hr[hriidx, c("AttritionProb_Year1", "AttritionProb_Year2", "AttritionProb_Year3",
                      "AttritionProb_Year4", "AttritionProb_Year5")]
plot(1, type = "n", xlim = c(1, ncol(hridv)), ylim = range(hridv), xlab = "Year", ylab = "Probability")
title("100 High-Risk Employee Turnoever Trajectory")
for (i in 1:nrow(hridv)) {
  lines(1:ncol(hridv), hridv[i, ], col = i, type = "l")
}
legend("topright", legend = hr[hriidx, "EmployeeNumber"], col = 1:nrow(hridv), lty = 1, title = "Employee Number")

# Save the model
saveRDS(model_v2, "model.rds")
saveRDS(hr, "transformed_data.rds")
