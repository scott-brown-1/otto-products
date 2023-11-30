library(tidyverse)

source('./scripts/constants.R')

load_data <- function(which='train',factor_cutoff=10){
  .train <- vroom::vroom(TRAIN_PATH)
  .test <- vroom::vroom(TEST_PATH)
  
  ## Fix dtypes
  for(col in colnames(.train)){
    if(length(unique(.train[[col]])) <= factor_cutoff){
      .train[[col]] <- as.factor(.train[[col]])
      
      if(col %in% colnames(.test)){
        .test[[col]] <- as.factor(.test[[col]])
      }
      
      print(paste('Changed column',col,'to factor.'))
    }
  } 
  
  ## Add row sums
  # .train['row_sum'] <- .train %>%
  #   select_if(is.numeric) %>%
  #   rowSums()
  # 
  # .test['row_sum'] <- .test %>%
  #   select_if(is.numeric) %>%
  #   rowSums()
  # 
  #train['row_sum'] <- tr
  
  #train['target'] <- factor(train$)

  return(list(
    'train'=.train,
    'test'=.test
  ))
}
