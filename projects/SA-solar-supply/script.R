
library(forecast)

rm(list = ls())
energy <- read.csv("extracted.csv")
energy <- energy[energy["Name"] == "BNGSF1"] 

autoplot(energy$Energy)