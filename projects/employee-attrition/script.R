
# Importing data
setwd("Downloads/shuuheialb.github.io/projects/employee-attrition")
rm(list = ls())
DATA <- read.csv("hr_data.csv")

# Save some test and validation data
library(rsample)
set.seed(100)
SPLIT <- initial_validation_split(DATA, c(0.75, 0.15))
TEST <- testing(SPLIT)
hr_valid <- validation(SPLIT)
hr <- training(SPLIT)
hr_full <- rbind(hr, hr_valid)


# Check for incorrect/problematic entries
# First few rows
#library(DataExplorer)
library(dplyr)
library(ggplot2)
library(reshape2)
#head(hr)
# All columns' types and freq/dist (summary and plots)
#str(hr)
#summary(hr)
#plot_bar(hr, title = "Categorical Variable Frequency", nrow = 2, ncol = 2)
#plot_histogram(hr, title = "Numerical Variable Distribution", nrow = 3, ncol = 3)
# Info about missing values and outliers
#plot_intro(hr, title = "Missing Values")


# Constant vectors:
# Formatted columns
cat_cols <- c("BusinessTravel", "Department", "Gender", "EducationField", "JobRole", "MaritalStatus")
bool_cols <- c("Attrition", "Over18", "OverTime")
char_to_bool <- function (col) { col %in% c("Yes", "Y", TRUE) }
# Irrelevant columns
is_single_col <- function (col) { length(unique(col)) == 1 }
invalid_unif_cols <- c("DailyRate", "MonthlyRate", "HourlyRate")
# Correlated columns
coupled_temporal_cols <- c("Age", "TotalWorkingYears", "YearsWithCurrManager", "YearsInCurrentRole", "YearsSinceLastPromotion")
high_corr_cols <- c("JobLevel")


library(survival)
library(randomForestSRC)
library(SurvMetrics)
library(pec)

# === Models

TIME <- "YearsAtCompany"
STATUS <- "Attrition"
ID <- "EmployeeNumber"

coxph_model <- function (df) {
  formula <- as.formula(paste("Surv(", TIME, ", ", STATUS, ") ~ ."))
  model <- coxph(data = df %>% select(-any_of(ID)), formula = formula, method = "efron", x = TRUE)
  return(model)
}
rsf_model <- function (df, params = NULL) {
  if (is.null(params)) {
    params <- list(ntree = 100, mtry = NULL)
  }
  formula <- as.formula(paste("Surv(", TIME, ", ", STATUS, ") ~ ."))
  model <- rfsrc(data = df %>% select(-any_of(ID)), formula = formula,
                 ntree = params$ntree, mtry = params$mtry)
}
model_f <- function (mode, params = NULL) { if (mode == "coxph") coxph_model
                                            else if (mode == "rsf") function(df) { rsf_model(df, params = params) }
                                            else NULL }

# === Feature selection

# 1. Cox score
coxph_score <- function (df, model) { # ignore the model
  features <- names(df)[!(names(df) %in% c(TIME, STATUS))]
  scores <- sapply(features, function (col){
    summary(coxph_model(df[c(TIME, STATUS, col)]))$concordance[1]
  })
  names(scores) <- features
  return(scores)
}
coxph_details <- function (model) {
  return(summary(model)$coefficients[, c("exp(coef)", "coef", "se(coef)")])
}
# 2. RSF Importance
rsf_importance <- function (df, model) {
  return(predict(model, data = df, importance = TRUE)$importance)
}
feature_rank_f <- function (mode) { if (mode == "coxph") coxph_score else if (mode == "rsf") rsf_importance else NULL }
# ===
print_feature_rank <- function (df, mode) {
  if (mode == "coxph") {
    print("Univariate Cox score:")
  } else {
    print("Variable importance:")
  }
  res <- feature_rank_f(mode)(df, model_f(mode)(df))
  print(res[order(res, decreasing = TRUE)])
}

# === Accuracy metrics

# SurvMetrics variables incompatible with custom functions:
#   model after df then tested after df2
survMetrics_wrapper <- function (df, df2, mode, metric_f) {
  tmp <- list(time = TIME, status = STATUS)
  TIME <<- "time"
  STATUS <<- "status"
  rename_survmetrics <- function (df) {
    names(df)[names(df) == tmp$time] <- TIME
    names(df)[names(df) == tmp$status] <- STATUS
    return(df)
  }
  res <- metric_f(model_f(mode)(rename_survmetrics(df)), rename_survmetrics(df2), t_star = -1)
  TIME <<- tmp$time
  STATUS <<- tmp$status
  return(res)
}
# 1. Concordance index
c_index <- function (df, df2, mode) {
  return(survMetrics_wrapper(df, df2, mode, SurvMetrics::Cindex))
}
# 2. Brier score: SOON
brier_score <- function (df, df2, mode) {
  return(survMetrics_wrapper(df, df2, mode, SurvMetrics::Brier))
}
# Legacy code from pec:
plot_pec <- function (df, model) { # split???
  formula <- as.formula(paste("Surv(", TIME, ", ", STATUS, ") ~ ."))
  pred_error <- pec(model, data = df, splitmethod = "cv10", formula = formula, cens.model = "marginal", reference = FALSE)
  plot(pred_error, xlim = c(0, 10), ylim = c(0, 0.25)) # 0.25 is the worst case scenario (random model)
  title("Prediction Error Curve")
}
# ===
print_c_index <- function (df, df2, mode) {
  print("Concordance index:")
  print(c_index(df, df2, mode))
}

# === Cross-validation

cross_val <- function (df, mode, feature_num = 1, params = NULL, k = 5) {
  random_index <- sample(1:nrow(df))
  performance_score1_vec <- numeric(k)
  for (fold in 1:k) {
    prev_end <- round((fold-1)/k * nrow(df))
    size <- round(fold * nrow(df)/k) - round((fold-1) * nrow(df)/k)
    idx <- random_index[(prev_end+1):(prev_end+size)]
    test <- df[idx, ]
    train <- df[-idx, ]
    
    # Preprocessing
    tt <- prepare(train, test, mode)
    train <- tt[[1]]
    test <- tt[[2]]
    
    # Initial model
    model1 <- model_f(mode, params)(train)
    feature_rank1 <- feature_rank_f(mode)(test, model1)
    feature_rank1 <- feature_rank1[order(feature_rank1, decreasing = TRUE)]
    
    # Correlated columns removal
    feature_redundancy <- corr_table(train, limit = 0.8)
    for (i in 1:nrow(feature_redundancy)) {
      # Retain higher rank, if any
      x <- feature_redundancy[i, "Var1"]
      y <- feature_redundancy[i, "Var2"]
      if (x %in% feature_rank1 && y %in% feature_rank1) {
        x_rank <- which(feature_rank1 == x)
        y_rank <- which(feature_rank1 == y)
        feature_rank1 <- feature_rank1[-max(x_rank, y_rank)]
      }
    }
    
    # Next model
    feature_rank2 <- names(feature_rank1)[1:min(feature_num, length(names(feature_rank1)))]
    #print("Choosing features ...")
    #print(feature_rank2)
    train2 <- train[c(TIME, STATUS, feature_rank2)]
    test2 <- test[c(TIME, STATUS, feature_rank2)]
    performance_score1 <- c_index(train2, test2, mode)
    # Brier score soon
    
    performance_score1_vec[fold] <- performance_score1
  }
  
  # !!! Box plot soon
  print(paste0("Performance score via concordance index, for feature number = ", feature_num, ":"))
  print(paste0("Mean: ", mean(performance_score1_vec)))
  print(performance_score1_vec)
  print(paste0("SE: ", sd(performance_score1_vec)))
}

# === Preprocessing: very dataset-specific so no need for TIME/STATUS/ID here

prepare <- function (df, df2, mode = "") {
  df <- prepare_gen(df)
  df2 <- prepare_gen(df2)
  
  # Train set additional process: intern and outlier removal
  df <- df %>%
    mutate(IsIntern = (MonthlyIncome < 2000)) %>%
    mutate(iqr = IQR(YearsAtCompany), 
           q1 = quantile(YearsAtCompany, probs = c(0.25)) - 3*iqr,
           q3 = quantile(YearsAtCompany, probs = c(0.75)) + 3*iqr,
           IsOutlier = between(YearsAtCompany, q1, q3)) %>%
    select(-c(iqr, q1, q3))
  df_train <- df %>% filter(!(IsIntern & IsOutlier)) %>% select(-c(IsIntern, IsOutlier))
  df_removed <- df %>% filter(IsIntern & IsOutlier) %>% select(-c(IsIntern, IsOutlier))
  
  # Assumption check for CoxPH model
  if (mode == "coxph") {
    ph_obj <- df_train %>%
      select(-c(Department, JobRole)) %>%
      coxph_model %>% proportional_hazard_table
    non_ph_cols <- names(which(ph_obj$table[, "p"] < 0.01))
    non_ph_cols <- non_ph_cols[!(non_ph_cols == "GLOBAL")]
    for (col in non_ph_cols) { # There should be a better Tidyverse syntax
      #inter_col <- paste0("Inter_TIME_", col)
      #df_train[inter_col] <- df_train[TIME] * df_train[col]
      #df2[inter_col] <- df2[TIME] * df2[col]
      #df_removed[inter_col] <- df_removed[TIME] * df_removed[col]
    }
  }
  return(list(df_train, df2, df_removed))
}
prepare_gen <- function (df) {
  df %>%
    mutate_at(cat_cols, factor) %>%
    mutate_at(bool_cols, char_to_bool) %>%
    select(-where(is_single_col)) %>%
    select(-any_of(invalid_unif_cols)) %>%
    mutate(NotWorkingYears = Age - TotalWorkingYears) %>%
    mutate(YearsAtOtherCompanies = TotalWorkingYears - YearsAtCompany) %>%
    select(-any_of(coupled_temporal_cols)) %>%
    select(-any_of(high_corr_cols))
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
coxph_cols <- c("Attrition", "YearsAtCompany", "EmployeeNumber", "MonthlyIncome", "JobRole", "OverTime")
rsf_cols <- c("Attrition", "YearsAtCompany", "EmployeeNumber", "OverTime", "MonthlyIncome", "StockOptionLevel", "TrainingTimesLastYear",
              "EnvironmentSatisfaction", "JobRole", "NumCompaniesWorked", "NotWorkingYears", "WorkLifeBalance")
TT_c <- prepare(hr_full, TEST, mode = "coxph")
TT_r <- prepare(hr_full, TEST, mode = "rsf")
TRAIN_c <- TT_c[[1]][coxph_cols]
TRAIN_r <- TT_r[[1]][rsf_cols]
coxph_final_model <- coxph_model(TRAIN_c)
rsf_final_model <- rsf_model(TRAIN_r)


# === Plotly Visualisation: Legacy Code

# Preparing the test data, rejoining training with ousted data
TEST_c <- TT_c[[2]][coxph_cols]
TEST_r <- TT_r[[2]][rsf_cols]
hr_full_c <- rbind(TRAIN_c, TT_c[[3]][coxph_cols])
hr_full_r <- rbind(TRAIN_r, TT_r[[3]][rsf_cols])

# Appending predictions
years = 5
rsf_risk <- predict(rsf_final_model, hr_full_r)$chf[, 1 + (0:years)] # matrix
rownames(rsf_risk) <- rownames(hr_full_r) # For Plotly
hr_full_r["Hazard_RSF_in_5yr"] <- rsf_risk[, 6]
hr_full_c["HazardRatio_CoxPH"] <- predict(coxph_final_model, hr_full_c, type = "risk")

# Table sorted by hazards
cutoff = 20
sorted <- function (df, key_col) {
  df <- df[df["Attrition"] == FALSE, ]
  return(df[order(df[, key_col], decreasing = TRUE), ][1:cutoff, ])
}
hr_sorted_rsf <- sorted(hr_full_r, "Hazard_RSF_in_5yr")
hr_sorted_coxph <- sorted(hr_full_c, "HazardRatio_CoxPH")
head(hr_sorted_rsf)
head(hr_sorted_coxph)

# Plotly implementation
suppressMessages(library(plotly))

fig <- plot_ly()

# --- Adding traces
# RSF
for (rank in 1:cutoff) {
  employee_row <- hr_sorted_rsf[rank, ]
  employee_row_idx_str <- rownames(employee_row)
  fig <- fig %>% add_trace(x = 0:years,
                           y = rsf_risk[employee_row_idx_str, ],
                           name = employee_row$EmployeeNumber,
                           type = "scatter", mode = "lines+markers", # Every time you put type, it assign DOMNum = 0
                           marker = list(size = 6),
                           line = list(shape = "spline", width = 2),
                           hovertemplate = paste0("<b>", rank, ". Employee ", employee_row$EmployeeNumber, "</b>",
                                                  "<br>Year: %{x}",
                                                  "<br>Cumulative risk: %{y:.2f}"),
                           hoverlabel = list(font_size = 16),
                           legendrank = rank,
                           legendgroup = employee_row$EmployeeNumber,
                           showlegend = TRUE)
}

# CoxPH
fig <- fig %>% add_trace(data = hr_sorted_coxph,
                         x = 1:cutoff,
                         y = ~HazardRatio_CoxPH,
                         type = "bar",
                         marker = list(color = 1:cutoff, colorscale = list(c(0, "#64a1f4"), c(1, "#bfe6b5"))), # blue-green
                         hovertemplate = paste0("<b>%{x}. Employee ", hr_sorted_coxph$EmployeeNumber, "</b>",
                                                "<br>Risk ratio: %{y:.2f}"),
                         hoverlabel = list(bgcolor = "white", font_size = 16),
                         showlegend = FALSE,
                         visible = FALSE)


# --- Decorators
# Including the baseline
avg_cum_hazard <- -log(1-mean(hr_full_r[, "Attrition"]))
baseline <- list(
  type = "line", x0 = 0, x1 = 5, y0 = avg_cum_hazard, y1 = avg_cum_hazard,
  line = list(dash = "dash", width = 4, color = "#82A0D8")
)
fig <- fig %>% add_text(
  x = 0.3, y = 0.22,
  text = "Average hazard", textfont = list(family = "sans serif", size = 12),
  showlegend = FALSE
)

# Helper functions to group CHF based on variables
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
    color_num <- 8
    color_palletes <- brewer.pal(color_num, "Set2")
    var_factors <- as.numeric(factor(hr_sorted_rsf[[var]]))
    button <- list(
      method = "restyle",
      label = var,
      args = list(list( # Please read Plotly.js for detailed documentation (not R/Python)
        line.color = color_palletes[var_factors %% color_num + 1],
        marker.color = color_palletes[var_factors %% color_num + 1],
        legendrank = if(var == "EmployeeNumber") 1:cutoff else var_factors,
        legendgroup = hr_sorted_rsf[[var]],
        showlegend = 1:cutoff %in% unique_index(var_factors),
        name = as.character(hr_sorted_rsf[[var]])
      ), 1:cutoff - 1)
    )
    return(button)
  })
}


# --- Setting layout
rsf_layout = list( 
  title = "Top 20 High-Risk Employees, according to RSF's Cumulative Risk in Five Years",
  shapes = list(baseline),
  xaxis = list(title = "Year", range = c(0, 5), zerolinecolor = 'white'),
  yaxis = list(title = "Accummulated Risk over Time", range = c(0, 1.2), zerolinecolor = 'white'),
  `annotations[1].visible` = TRUE,
  `updatemenus[1].visible` = TRUE
)
coxph_layout = list(
  title = "Top 20 High-Risk Employees, according to CoxPH's Risk Ratio",
  shapes = list(),
  xaxis = list(title = "Employee ID", tickmode = "array", tickvals = 1:cutoff,
               ticktext = hr_sorted_coxph$EmployeeNumber, zerolinecolor = 'white'),
  yaxis = list(title = "Risk Ratio", zerolinecolor = 'white'),
  `annotations[1].visible` = FALSE,
  `updatemenus[1].visible` = FALSE
)
fig <- fig %>% layout( # do.call soon
  title = "Top 20 High-Risk Employees, according to RSF's Cumulative Risk in Five Years",
  shapes = list(baseline),
  xaxis = list(title = "Year", range = c(0, 5), zerolinecolor = 'white'),
  yaxis = list(title = "Accummulated Risk over Time", range = c(0, 1.2), zerolinecolor = 'white'),
  margin = list(l = 50, t = 50, b = 50, r = 50),
  plot_bgcolor = '#e5ecf6',
  annotations = list(
    list(
      xanchor = "left", yanchor = "top", x = -0.5, y = 1, xref = "paper", yref = "paper", 
      text = "Method", align = "left", showarrow = FALSE
    ), list(
      xanchor = "left", yanchor = "top", x = -0.5, y = 0.8, xref = "paper", yref = "paper", 
      text = "Based on", align = "left", showarrow = FALSE
    )
  ),
  updatemenus = list(
    list(
      xanchor = "left", yanchor = "top", x = -0.5, y = 0.95, xref = "paper", yref = "paper",
      showactive = TRUE,
      buttons = list(
        list(
          label = "Random Survival Forest",
          method = "update",
          args = list(list(visible = c(rep(TRUE, cutoff), FALSE, TRUE)), rsf_layout, 1:(cutoff+2) - 1)
        ),
        list(
          label = "Cox Proportional Hazard",
          method = "update",
          args = list(list(visible = c(rep(FALSE, cutoff), TRUE, FALSE)), coxph_layout, 1:(cutoff+2) - 1)
        )
      )
    ), list(
      xanchor = "left", yanchor = "top", x = -0.5, y = 0.75, xref = "paper", yref = "paper",
      showactive = TRUE,
      buttons = create_buttons(c("EmployeeNumber", "EnvironmentSatisfaction", "JobRole", "MonthlyIncome", "NumCompaniesWorked",
                                 "NotWorkingYears", "OverTime", "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance"))
    )
  )
)

# Build the HTML widget, only for the case of nbviewer
# Otherwise, simply the line `fig` will suffice
#htmlwidgets::saveWidget(as_widget(fig), "plot.html")
#IRdisplay::display_html("<iframe seamless src='plot.html' width=1000, height=600></iframe>")
fig