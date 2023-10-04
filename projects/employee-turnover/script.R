
# Import & clean
rm(list = ls())
setwd("Downloads/shuuheialb.github.io/projects/employee-turnover")
hr <- read.csv("emp_attrition.csv")

remove_cols <- function(df, cols) df[, !(names(df) %in% cols)]
hr <- remove_cols(hr, c("EmployeeNumber"))
single_value_cols <- names(hr)[sapply(hr, function (col) length(unique(col)) == 1)]
hr <- remove_cols(hr, single_value_cols)
cat_cols <- c("Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
hr[cat_cols] <- lapply(hr[cat_cols], factor)
summary(hr)
hr$Attrition <- ifelse(hr$Attrition == "Yes", TRUE, FALSE)

# Model building
library(randomForestSRC)
library(survival)
library(pec, warn.conflicts = FALSE)
library(ggplot2)
library(reshape2)

# Model
rsf_model <- function (df) {
  model <- rfsrc(Surv(YearsAtCompany, Attrition) ~ ., data = df, ntree = 100)
  return(model)
}

# For tweaking model
rsf_pec <- function(df, model) {
  pred_error <- pec(model, data = df, formula = Surv(YearsAtCompany, Attrition) ~ .,
                    splitMethod = "cv10", cens.model = "marginal")
  return(pred_error)
}
rsf_var_importance <- function (df, model) {
  importance <- predict(model, importance = TRUE)$importance
  sorted_importance <- importance[order(importance, decreasing = TRUE)]
  return(sorted_importance)
}
plot_correlation <- function (df, model) {
  cor_matrix <- cor(df[, !(names(df) %in% cat_cols)]) # Warning: global cat_cols
  melted_cor_matrix <- melt(cor_matrix)
  ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "lightblue", high = "darkblue", guide = "colorbar") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
}
# No need to cross validate the RSF, but maybe for backup
train_test_generate <- function (df, seed = 123, proportion = 0.7) {
  set.seed(seed)
  size <- round(proportion * nrow(df))
  idx <- sample(seq_len(nrow(df)), size = size, replace = FALSE)
  return(list(df[idx, ], df[-idx, ]))
}

# Result
rsf_turnover_probability <- function (df, model) {
  return(predict(model, df)$chf)
}

# First iteration
model <- rsf_model(hr)
plot(rsf_pec(hr, model), xlim = c(0,30))
print(rsf_var_importance(hr, model))
plot_correlation(hr, model)