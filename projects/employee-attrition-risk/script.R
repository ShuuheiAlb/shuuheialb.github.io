
# Import & clean
setwd("Downloads/shuuheialb.github.io/projects/employee-attrition")
rm(list = ls())
hr <- read.csv("hr_data.csv")

# Filter columns
cols <- names(hr)
cols <- cols[cols != "EmployeeNumber"]
single_value_cols <- names(hr)[sapply(hr, function (col) length(unique(col)) == 1)]
cols <- cols[!(cols %in% single_value_cols)]

# Convert column data types
cat_cols <- c("BusinessTravel", "Department", "EducationField", "JobRole")
hr[cat_cols] <- lapply(hr[cat_cols], factor)
bool_cols <- c("Attrition", "Gender", "MaritalStatus", "OverTime")
hr[bool_cols] <- lapply(hr[bool_cols], function (col) ifelse(col == "Yes", TRUE, FALSE))

# Adding columns
hr["NotWorkingYears"] <- hr["Age"] - hr["TotalWorkingYears"]
hr["YearsAtOtherCompanies"] <- hr["TotalWorkingYears"] - hr["YearsAtCompany"]
# Removing outliers
outlier_index <- function(df, col) {
  vec <- df[[col]]
  qnt <- quantile(vec, probs = c(0.25, 0.75))
  iqr <- qnt[2] - qnt[1]
  min <- qnt[1] - 3 * iqr
  max <- qnt[2] + 3 * iqr
  return(which(vec < min | vec > max))
}
hr_train <- hr[-outlier_index(hr, "YearsAtCompany"), ]

library(survival)
library(randomForestSRC)
library(pec, warn.conflicts = FALSE)
library(ggplot2)
library(reshape2)
library(viridis)

# Models
coxph_model <- function (df) {
  model <- coxph(data = df, formula = Surv(YearsAtCompany, Attrition) ~ ., method = "breslow", x = TRUE)
  return(model)
}
rsf_model <- function (df) {
  model <- rfsrc(Surv(YearsAtCompany, Attrition) ~ ., data = df, ntree = 100)
  return(model)
}

# Cross-validation functions
train_test_generate <- function (df, proportion = 0.7) {
  size <- round(proportion * nrow(df))
  idx <- sample(seq_len(nrow(df)), size = size, replace = FALSE)
  return(list("train" = df[idx, ], "test" = df[-idx, ]))
}
cross_val <- function (df, model_f, metric_f, k = 20) {
  for (fold in 1:k) {
    separate <- train_test_generate(df)
    train <- separate$train
    test <- separate$test
    model <- model_f(train)
    if (!(exists("metric_tot"))) {
      metric_tot <- metric_f(test, model)
    } else {
      metric_tot <- metric_tot + metric_f(test, model)
    }
  }
  metric_avg <- metric_tot/k
  return(metric_avg[order(abs(metric_avg), decreasing = TRUE)])
}

# Validation metrics
# 1. Feature rank
# ----- Cox score
coxph_score <- function (df, model) {
  return(summary(model)$coefficients[, "z"]) # the test data is indeed not used
}
print_coxph_var_rank <- function (df) {
  print("Average univariate Cox score:")
  print(cross_val(df, coxph_model, coxph_score))
}
coxph_slope <- function (df, model) {
  return(summary(model)$coefficients[, "coef"])
}
# ----- Importance
rsf_importance <- function (df, model) {
  importance <- predict(model, df, importance = TRUE)$importance
  return(importance)
}
print_rsf_var_rank <- function (df) {
  print("Average variable importance:")
  print(cross_val(df, rsf_model, rsf_importance))
}
# 2. C-index
print_c_index <- function(df, model_f) {
  print("Average concordance index:")
  print(cross_val(df, model_f, function (sub_df, model) {
    unlist(cindex(model, formula = Surv(YearsAtCompany, Attrition) ~ ., data = sub_df)$AppCindex)
  }))
}
print_coxph_var_rank <- function (df, k = 20) {
  print("Average univariate Cox score:")
  print(cross_val(df, coxph_model, coxph_score, k))
}
# 3. PEC
plot_pec <- function(df, model_f) {
  suppressMessages(pred_error <- pec(model_f(df), data = df, formula = Surv(YearsAtCompany, Attrition) ~ .,
                                     splitMethod = "cv10", cens.model = "marginal", reference = FALSE))
  plot(pred_error, xlim = c(0, 10), ylim = c(0, 0.25)) # 0.25 is the worst case scenario (random model)
  title("Prediction Error Curve")
}

# Assumption functions
check_proportional_hazard <- function(model) {
  print(cox.zph(model))
}
plot_correlation <- function (df, limit = -1) {
  # Only numerical (unless categories can be continuously extended)
  num_cols <- sapply(df, is.numeric)
  bool_cols <- sapply(df, is.logical)
  df2 <- df[, num_cols | bool_cols]
  
  # Only select the high-correlated columns
  cor_matrix <- cor(df2)
  high_cor_pairs <- which(cor_matrix >= limit & cor_matrix < 1, arr.ind = TRUE)
  high_cor_cols <- colnames(df2)[high_cor_pairs[, 1]]
  ggplot(data = melt(cor_matrix[high_cor_cols, high_cor_cols]), aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() + scale_fill_viridis() +
    xlab("") + ylab("") + ggtitle("Correlation Map") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
          plot.title = element_text(hjust = 0.5))
}

# Final iteration
rsf_cols <- c("Attrition", "YearsAtCompany", "JobLevel", "OverTime", "StockOptionLevel",
          "JobRole", "EnvironmentSatisfaction", "NotWorkingYears")
coxph_cols <- c("Attrition", "YearsAtCompany", "OverTime", "JobInvolvement", "DistanceFromHome",
              "JobSatisfaction", "BusinessTravel", "EnvironmentSatisfaction", "MaritalStatus",
              "Gender")
rsf_final_model <- rsf_model(hr[rsf_cols])
coxph_final_model <- coxph_model(hr[coxph_cols])
cols <- c("Attrition", "YearsAtCompany", "MonthlyIncome", "OverTime", "Age",
          "JobRole", "NumCompaniesWorkedPerYear", "StockOptionLevel")

# Prediction
hr["HazardRatio_CoxPH"] <- predict(coxph_final_model, hr, type = "risk")
# coxph_risk <- exp(predict(coxph_final_model, hr, type = "lp")) %*% t(basehaz(coxph_final_model)["hazard"])

years = 5
rsf_risk <- predict(rsf_final_model, hr)$chf[, 1:(years+1)] # matrix
rsf_survival_probs <- predict(rsf_final_model, hr)$survival[, 1:(years+1)]
hr["Hazard_RSF_Current"] <- rsf_risk[, 1]
hzrsf_cols <- sapply(0:years, function(i) {paste0("Hazard_RSF_", i)})
hr[hzrsf_cols] <- rsf_risk[, 1 + (0:years)]

preview_cols <- c("EmployeeNumber", coxph_cols, rsf_cols, "HazardRatio_CoxPH", hzrsf_cols)

# Table sorted by Cox Hazard Ratio
cutoff = 50
sorted <- function (df, key_col) {
  df <- df[df["Attrition"] == FALSE, ]
  return(row.names(df)[order(df[, key_col], decreasing = TRUE)][1:cutoff])
}
hr_sorted_coxph <- hr[sorted(hr, "HazardRatio_CoxPH"), ]
head(hr_sorted_coxph[, preview_cols])

#hridv <- hr[hr_sorted_coxph(hr, "Hazard_RSF"), preview_cols]
#plot(1, type = "n", xlim = c(0, 5), ylim = c(0, 1), xlab = "Year", ylab = "Probability")
#title("High-Risk Employee Turnoever Trajectory")
#for (i in 1:cutoff) {
#  lines(0:years, hridv[i, ap_cols], col = i, type = "l")
#}
#legend("topleft", legend = hridv[, "EmployeeNumber"], col = 1:nrow(hridv), lty = 1, title = "Employee Number")

# Plotly: Options to group them based on variables?

suppressMessages(library(plotly))

group <- "EmployeeNumber"
hr_sorted_rsf <- hr[sorted(hr, "Hazard_RSF_Current"), ]

# Adding trajectory risks
fig <- plot_ly(type = "scatter", mode = "lines+markers")
for (rank in 1:cutoff) {
  employee_row_num <- as.numeric(rownames(hr_sorted_rsf)[rank])
  employee_row <- hr_sorted_rsf[rank, ]
  x <- 0:(ncol(rsf_risk)-1)
  y <- rsf_risk[employee_row_num, ]
  fig <- fig %>% add_trace(x = x,
                           y = y,
                           name = employee_row["EmployeeNumber"],
                           marker = list(size = 4),
                           line = list(shape = "spline", width = 1),
                           hoverinfo = "text",
                           text = paste0("<i>#", rank, ": Employee ", employee_row["EmployeeNumber"], "</i>",
                                         "\nYear: ", x,
                                         "\nScore: ", y),
                           legendgroup = employee_row["EmployeeNumber"],
                           showlegend = TRUE
  )
}

# Including the baseline
avg_cum_hazard <- -log(1-mean(hr[, "Attrition"]))
baseline <- list(
  type = "line", x0 = 0, x1 = 5, y0 = avg_cum_hazard, y1 = avg_cum_hazard,
  line = list(dash = "dash", width = 4, color = "#82A0D8")
)
fig <- fig %>% layout(
  shapes = list(baseline)
) #%>% add_text(
#  showlegend = FALSE, x = 0.25, y = 0.22, text = "Average hazard", 
)

# Adding options to group them based on variables
library(RColorBrewer)
unique_index <- function(x) {
  unique_vals <- unique(x)
  first_occur <- integer(length(unique_vals))
  for (i in 1:length(unique_vals)) {
    first_occur[i] <- which(x == unique_vals[i])[1]
  }
  return(first_occur)
}
create_buttons <- function(vars) {
  lapply(vars, function(var) {
    color_palletes <- brewer.pal(12, "Set3")
    var_factors <- as.numeric(factor(hr_sorted_rsf[[var]]))
    button <- list(
      method = "restyle",
      label = var,
      args = list(list( # Some elements are just "double-list". This changes style to every DOM elements
        line.color = color_palletes[var_factors],
        marker.color = color_palletes[var_factors],
        legendgroup = factor(hr_sorted_rsf[[var]]),
        showlegend = 1:cutoff %in% unique_index(var_factors),
        name = factor(hr_sorted_rsf[[var]]),
        trace = 1:50
      ))
    )
    return(button)
  })
}

# Setting layout
fig <- fig %>% layout(
  title = "High-Risk Employee Attrition Trajectory",
  xaxis = list(title = "Year", range = c(0, 5), zerolinecolor = '#ffff'),
  yaxis = list(title = "Accummulated Risk until Year _", range = c(0, 1.5), zerolinecolor = '#ffff'),
  plot_bgcolor = '#e5ecf6',
  updatemenus = list(
    list(
      y = 0.9,
      showactive = TRUE,
      buttons = create_buttons(c("EmployeeNumber", "BusinessTravel", "EnvironmentSatisfaction", "Gender",
                                 "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "NumCompaniesWorked",
                                 "OverTime", "StockOptionLevel"))
    )
  ),
  annotations = list(
    list(
      text = "Based on", x = -0.3, xref = "paper", y = 1, yref = "paper", align = "left", showarrow = FALSE
    )
  )
)

fig