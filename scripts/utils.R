library(tidyverse)

source('./scripts/constants.R')

load_train <- function(){
  train <- vroom::vroom(TRAIN_PATH)
}

load_test <- function(){
  train <- vroom::vroom(TEST_PATH)
}


# apply_factors <- function(){
#   
# }
# 
# df <- load_train()
# 
# for(col in colnames(df)){
#   if(length(unique(df[col])) == 2){
#     print('A')
#   }
# }
