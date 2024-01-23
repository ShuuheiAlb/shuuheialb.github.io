
# Importing data
setwd("Downloads/shuuheialb.github.io/projects/employee-attrition")
rm(list = ls())
DATA <- read.csv("hr_data.csv")

# Save some test and validation data
library(rsample)
set.seed(100)
SPLIT <- initial_validation_split(DATA, c(0.75, 0.15))
TEST <- testing(SPLIT)
VALID <- validation(SPLIT)
hr <- training(SPLIT)
hr_full <- rbind(hr, VALID)


# Check for incorrect/problematic entries
#head(hr, 10)
#nrow(hr[duplicated(hr), ])
#sum(is.na(hr))
#summary(hr)
#sapply(hr, function(x) {
#  unique_vals <- sort(unique(x))
#  if (length(unique_vals) <= 30) return(unique_vals)
#  return(c(unique_vals[1:30], "etc"))
#})


# Constant vectors:
# Formatted columns
cat_cols <- c("BusinessTravel", "Department", "Gender", "EducationField", "JobRole", "MaritalStatus")
bool_cols <- c("Attrition", "Over18", "OverTime")
# Irrelevant columns
unique_id_col <- "EmployeNumber"
single_value_cols <- names(hr)[sapply(hr, function (col) length(unique(col)) == 1)]
invalid_unif_cols <- c("DailyRate", "MonthlyRate", "HourlyRate")
# Correlated columns
coupled_cols <- c("Age", "TotalWorkingYears", "YearsWithCurrManager", "YearsInCurrentRole", "YearsSinceLastPromotion")
high_cor_cols <- c("JobLevel")
# Added columns
new_time_cols <- c("NotWorkingYears", "YearsAtOtherCompanies")


library(survival)
library(randomForestSRC)
library(pec, warn.conflicts = FALSE)
library(DataExplorer)
library(ggplot2)
library(reshape2)
library(corrplot)

# === Models

coxph_model <- function (df) {
  model <- coxph(data = df, formula = Surv(YearsAtCompany, Attrition) ~ ., method = "breslow", x = TRUE)
  return(model)
}
rsf_model <- function (df) {
  model <- rfsrc(Surv(YearsAtCompany, Attrition) ~ ., data = df, ntree = 100)
  return(model)
}

# === Feature selection

# 1. Cox score
coxph_score <- function (df, model) { # ignore the model
  features <- names(df)[!(names(df) %in% c("Attrition", "YearsAtCompany"))]
  scores <- sapply(features, function (col){
    summary(coxph_model(df[c("YearsAtCompany", "Attrition", col)]))$concordance[1]
  })
  names(scores) <- features
  return(scores)
}
coxph_details <- function (model) {
  return(summary(model)$coefficients[, c("exp(coef)", "coef", "se(coef)")])
}
# 2. RSF Importance
rsf_importance <- function (df, model) {
  return(predict(model, df, importance = TRUE)$importance)
}
# ===
print_coxph_var_rank <- function (df) {
  print("Univariate Cox score:")
  res <- coxph_score(df, coxph_model(df))
  # SOON: 1D line
  print(res[order(res, decreasing = TRUE)])
}
print_rsf_var_rank <- function (df) {
  print("Variable importance:")
  res <- rsf_importance(df, rsf_model(df))
  # SOON: 1D line
  print(res[order(abs(res), decreasing = TRUE)])
}

# === Accuracy metrics

# 1. C-index
c_index <- function (df, model) {
  c_table <- cindex(model, formula = Surv(YearsAtCompany, Attrition) ~ ., data = df)
  if (unlist(c_table$Pairs) == 0) return(0) # In case of zero events
  return(unlist(c_table$AppCindex))
}
# 2. PEC
# ===
print_c_index <- function(df, model_f) {
  print("Concordance index:")
  print(c_index(df, model_f(df)))
}
plot_pec <- function(df, model_f) { # split???
  suppressMessages(pred_error <- pec(model_f(df), data = df, formula = Surv(YearsAtCompany, Attrition) ~ .,
                                     splitMethod = "cv10", cens.model = "marginal", reference = FALSE))
  plot(pred_error, xlim = c(0, 10), ylim = c(0, 0.25)) # 0.25 is the worst case scenario (random model)
  title("Prediction Error Curve")
}

# === Cross-validation

cross_val <- function (df, model_f, select_f, feature_num = 1, k = 5) {
  random_index <- sample(1:nrow(df))
  performance_score1_vec <- numeric(k)
  for (fold in 1:k) {
    prev_end <- round((fold-1)/k * nrow(df))
    size <- round(fold * nrow(df)/k) - round((fold-1) * nrow(df)/k)
    idx <- random_index[(prev_end+1):(prev_end+size)]
    test <- df[idx, ]
    train <- df[-idx, ]
    
    # General preprocessing
    train <- prepare(train)
    test <- prepare(test)
    
    # Training-data preprocessing
    train <- train[-outlier_index(train, "YearsAtCompany", iqr_coef = 2), ]
    
    # Model-specific preprocessing
    if (identical(model_f, coxph_model)) {
      # Assumption check
      cols <- colnames(train)
      focus_cols <- cols[!(cols %in% c("Department", "JobRole"))]
      pht <- proportional_hazard_table(model_f(train[focus_cols]))
      broken_ph_assumption_cols <- names(which(pht$table[, "p"] < 0.01))
      for (col in broken_ph_assumption_cols) {
        inter_col <= paste0("Inter_YC_", col)
        cols <- c(cols, inter_col)
        train[inter_col] <- train[col] * train["YearsAtCompany"]
        test[inter_col] <- test[col] * test["YearsAtCompany"]
      }
    } else {
      # Probational data treatment
      cols <- c(cols, "IsProbation")
      train["IsProbation"] <- train["MonthlyIncome"] <= 2000
      test["IsProbation"] <- test["MonthlyIncome"] <= 2000
    }
    
    # Initial model
    model1 <- model_f(train)
    feature_score <- select_f(test, model1)
    feature_rank <- feature_score[order(abs(feature_score), decreasing = TRUE)]
    
    # Correlated columns removal
    feature_redundancy <- corr_table(train, limit = 0.7)
    for (i in 1:nrow(feature_redundancy)) {
      # Retain higher rank, if any
      x <- feature_redundancy[i, "Var1"]
      y <- feature_redundancy[i, "Var2"]
      if (x %in% feature_rank && y %in% feature_rank) {
        x_rank <- which(feature_rank == x)
        y_rank <- which(feature_rank == y)
        feature_rank <- feature_rank[-max(x_rank, y_rank)]
      }
    }
    
    # Next model
    final_features <- names(feature_rank)[1:min(feature_num, length(names(feature_rank)))]
    #print("Choosing features ...")
    #print(final_features)
    model2 <- model_f(train[c("Attrition", "YearsAtCompany", final_features)])
    performance_score1 <- c_index(test, model2)
    # PEC variable soon
    
    performance_score1_vec[fold] <- performance_score1
  }
  
  print(paste0("Performance score via concordance index, for feature number = ", feature_num, ":"))
  # Box plot soon
  print(paste0("Mean: ", mean(performance_score1_vec)))
  print(performance_score1_vec)
  print(paste0("SE: ", sd(performance_score1_vec)))
}
# ===
prepare <- function (df) {
  df[cat_cols] <- lapply(df[cat_cols], factor)
  df[bool_cols] <- lapply(df[bool_cols], function (col) { ifelse(col %in% c("Yes", "Y"), TRUE, FALSE) })
  df["YearsAtOtherCompanies"] <- df["TotalWorkingYears"] - df["YearsAtCompany"]
  df <- df[!(names(df) %in% c("EmployeeNumber", single_value_cols, invalid_unif_cols, coupled_cols, high_cor_cols))]
  return(df)
}
outlier_index <- function (df, col, iqr_coef = 0) {
  vec <- df[[col]]
  qnt <- quantile(vec, probs = c(0.25, 0.75))
  iqr <- qnt[2] - qnt[1]
  min <- qnt[1] - iqr_coef * iqr
  max <- qnt[2] + iqr_coef * iqr
  return(which(vec < min | vec > max))
}
proportional_hazard_table <- function (model) {
  return(cox.zph(model))
}
corr_table <- function (df, limit = 0){
  # Convert categories to dummies
  df2 <- model.matrix(~ 0+., df)
  
  # Dropping entries
  corr <- cor(df2)
  corr[lower.tri(corr, diag = TRUE)] <- NA
  corr[abs(corr) == 1] <- NA
  
  # Convert to long format, sorted
  res <- melt(corr)
  res <- na.omit(res)
  res <- res[abs(res$value) > limit,]
  res <- res[order(res$value, decreasing = TRUE),]
  rownames(res) <- 1:nrow(res)
  return(res)
}



# Final iteration
hr_full <- prepare(hr_full)
hr <- prepare(hr)
rsf_cols <- c("Attrition", "YearsAtCompany", "OverTime", "MonthlyIncome", "IsProbation", "StockOptionLevel", 
              "EnvironmentSatisfaction", "JobRole", "NumCompaniesWorked", "NotWorkingYears")
coxph_cols <- c("Attrition", "YearsAtCompany", "MonthlyIncome", "JobRole", "OverTime")
rsf_final_model <- rsf_model(hr[rsf_cols])
coxph_final_model <- coxph_model(hr[coxph_cols])
cols <- c("Attrition", "YearsAtCompany", "MonthlyIncome", "OverTime", "Age",
          "JobRole", "NumCompaniesWorkedPerYear", "StockOptionLevel")

# Appending predictions
years = 5
rsf_risk <- predict(rsf_final_model, hr)$chf[, 1 + (0:years)] # matrix
rsf_survival_probs <- predict(rsf_final_model, hr)$survival[, 1 + (0:years)]
hr["Hazard_RSF_in_1yr"] <- rsf_risk[, 2]
hr["HazardRatio_CoxPH"] <- predict(coxph_final_model, hr, type = "risk")
preview_cols <- c("EmployeeNumber", coxph_cols, rsf_cols, "Hazard_RSF_in_1yr", "HazardRatio_CoxPH")

# Table sorted by hazards
cutoff = 20
sorted <- function (df, key_col) {
  df <- df[df["Attrition"] == FALSE, ]
  return(row.names(df)[order(df[, key_col], decreasing = TRUE)][1:cutoff])
}
hr_sorted_rsf <- hr[sorted(hr, "Hazard_RSF_in_1yr"), ]
hr_sorted_coxph <- hr[sorted(hr, "HazardRatio_CoxPH"), ]
head(hr_sorted_rsf[, preview_cols])
head(hr_sorted_coxph[, preview_cols])


# Plotly implementation
suppressMessages(library(plotly))

group <- "EmployeeNumber"
hr_sorted_rsf <- hr[sorted(hr, "Hazard_RSF_in_5_Years"), ]

# Adding RSH trajectory risks
fig <- plot_ly()
employee_row_num_wrt_ <- match(1:length(hr_sorted_rsf), hr_sorted_rsf)
for (rank in 1:cutoff) {
  employee_row_num <- as.numeric(rownames(hr_sorted_rsf)[rank])
  employee_row <- hr_sorted_rsf[rank, ]
  x <- 0:(ncol(rsf_risk)-1)
  y <- rsf_risk[employee_row_num, ] #soon
  fig <- fig %>% add_trace(x = x,
                           y = y,
                           name = employee_row["EmployeeNumber"],
                           type = "scatter", mode = "lines+markers", # every time you put type, it assign DOMNum = 0
                           marker = list(size = 4),
                           line = list(shape = "spline", width = 1),
                           hovertemplate = paste0("<i>#", rank, ": Employee ", employee_row["EmployeeNumber"], "</i>",
                                                  "\nYear: ", x,
                                                  "\nRisk score: ", y),
                           legendgroup = employee_row["EmployeeNumber"],
                           showlegend = TRUE
  )
}

# CoxPH
fig <- fig %>% add_trace(data = hr_sorted_coxph,
                         y = ~HazardRatio_CoxPH,
                         type = "bar",
                         marker = list(color = ~YearsAtCompany),
                         hovertemplate = paste0("Employee ", hr_sorted_coxph$EmployeeNumber,
                                                "\nHazard ratio: ", hr_sorted_coxph$HazardRatio_CoxPH),
                         visible = FALSE, showlegend = FALSE)


# --- Decorators
# Including the baseline
avg_cum_hazard <- -log(1-mean(hr[, "Attrition"]))
baseline <- list(
  type = "line", x0 = 0, x1 = 5, y0 = avg_cum_hazard, y1 = avg_cum_hazard,
  line = list(dash = "dash", width = 4, color = "#82A0D8")
)
fig <- fig %>% layout(
  shapes = list(baseline)
) %>% add_text(
  showlegend = FALSE, x = 0.3, y = 0.22, text = "Average hazard",
  textfont = list(family = "sans serif", size = 10) 
)

# Helper function to group RSF metrics based on variables
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
        legendgroup = hr_sorted_rsf[[var]],
        showlegend = 1:cutoff %in% unique_index(var_factors),
        name = as.character(hr_sorted_rsf[[var]])
      ), 1:cutoff - 1) # The DOMNum it applies to
    )
    return(button)
  })
}


# --- Setting layout
coxph_layout = list(
  title = "High-Risk Employees, according to Hazard Ratio",
  shapes = list(),
  xaxis = list(title = "Rank"),
  yaxis = list(title = "Hazard Ratio")
)
rsf_layout = list( 
  title = "High-Risk Employees: RSF's Accumulated Risk Trajectory",
  shapes = list(baseline),
  xaxis = list(title = "Year", range = c(0, 5), zerolinecolor = '#ffff'),
  yaxis = list(title = "Accummulated Risk until Year _", range = c(0, 1.5), zerolinecolor = '#ffff')
)
fig <- fig %>% layout(unlist(rsf_layout)) %>% layout(
  title = "High-Risk Employees: RSF's Accumulated Risk Trajectory",
  shapes = list(baseline),
  xaxis = list(title = "Year", range = c(0, 5), zerolinecolor = '#ffff'),
  yaxis = list(title = "Accummulated Risk until Year _", range = c(0, 1.5), zerolinecolor = '#ffff'),
  plot_bgcolor = '#e5ecf6',
  updatemenus = list(
    list(
      x = -0.1, y = 0.9, yref = "paper",
      showactive = TRUE,
      buttons = list(
        list(
          label = "Random Survival Forest",
          method = "update",
          args = list(list(visible = c(rep(TRUE, cutoff), FALSE, TRUE)), rsf_layout, 1:(cutoff+2) - 1)
        ),
        list(
          label = "Cox Proportional",
          method = "update",
          args = list(list(visible = c(rep(FALSE, cutoff), TRUE, FALSE)), coxph_layout, 1:(cutoff+2) - 1)
        )
      )
    ), list(
      x = -0.1, y = 0.6, yref = "paper",
      showactive = TRUE,
      buttons = create_buttons(c("EmployeeNumber", "EnvironmentSatisfaction", "JobLevel", "JobRole",
                                 "MonthlyIncome", "MaritalStatus", "OverTime", "StockOptionLevel", "WorkLifeBalance"))
    )
  ), annotations = list(
    list(
      x = -0.35, y = 1, xref = "paper", yref = "paper", 
      text = "Method", align = "left", showarrow = FALSE
    ), list(
      x = -0.35, y = 0.7, xref = "paper", yref = "paper", 
      text = "Based on", align = "left", showarrow = FALSE, visible = TRUE
    )
  )
)

fig