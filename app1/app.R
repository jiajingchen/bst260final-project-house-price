#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
ui <- fluidPage(
  # Put code here for a slider with n as the id,
  # 'This is a slider' as the label, and {value, min, max} = {1, 0, 100}
  sliderInput(inputId="n",label="This is a slider",value =1, min=0,max=100),
  plotOutput(outputId = "hist")
)
server <- function(input,output){ 
  output$hist <- renderPlot({
    hist(rnorm(input$n))
  })
}
shinyApp(ui=ui,server=server)
