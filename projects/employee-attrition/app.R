
library(shiny)

data <- readRDS("transformed_data.rds")
model <- readRDS("model.rds")
for (i in 1:5) {
  hr[, paste0("AttritionProb_Year", i)] <- predict(model, data)$chf[, i+1]
}

ui <- fluidPage(
  navbarPage("Employee Attrition Prediction", tabsetPanel(id = "tabs",
    # Prediction page
    tabPanel("Prediction",
             sidebarLayout(
               sidebarPanel(
                 textInput("employee_number", "Employee Number"),
                 actionButton("submit", "Submit")
               ),
               mainPanel(
                 textOutput("prediction_result"),
                 verbatimTextOutput("contributing_factors")
               )
             )),
    
      # Top Individuals page
      tabPanel("Top Individuals at Risk",
               mainPanel(
                 # Allow selection of employees
                 textInput("employee_number", "Employee Number"),
               )),
      
      # Insights/Recommendations page
      tabPanel("Insights/Recommendations",
               mainPanel(
                 # Display insights/recommendations here
               ))
  ))
)

server <- function(input, output) {
  # Function to find the employee in the prediction data
  find_employee <- function(n) {
    employee_row <- which(data$EmployeeNumber == n)
    if (employee_row == 0) {
      return(NULL)  # Employee not found
    } else {
      return(hr[employee_row, ])
    }
  }
  
  # Action when the submit button is clicked
  observeEvent(input$submit, {
    employee <- find_employee(input$employee_number)
    if (is.null(employee)) {
      output$prediction_result <- renderText("Employee not found")
      output$contributing_factors <- renderPrint(NULL)
    } else {
      output$prediction_result <- renderText({
        paste0("Probability of Attrition:", employee[c("AttritionProb_Year1", "AttritionProb_Year2", "AttritionProb_Year3",
              "AttritionProb_Year4", "AttritionProb_Year5")])
      })
      output$contributing_factors <- renderPrint({
        "soon"
      })
    }
  })
  
  # Additional server logic for other sections (validity level, insights, recommendations)
  # ...
}

# Finally
shinyApp(ui, server)

