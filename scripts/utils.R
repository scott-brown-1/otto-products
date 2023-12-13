library(tidyverse)

source('./scripts/constants.R')

load_data <- function(factor_cutoff=25, drop_n_worst=0){
  .train <- vroom::vroom(TRAIN_PATH)
  .test <- vroom::vroom(TEST_PATH)
  
  ## Drop worst columns as determined by high ANOVA residual SS
  vals <- tibble()
  
  for(col in colnames(.train)){
    if(col == 'target' | col=='id') next
    f <- formula(paste0(col,'~target'))
    tryCatch(
      {
        mod <- anova(lm(f, data=.train))
        #p <- as.data.frame(mod)$`Pr(>F)`[1]
        ss <- as.data.frame(mod)$`Sum Sq`[2]
        vals <- rbind(vals,c(col,ss))
      },
      error = function(e){print(e);next}
    )
  }
  
  worst_n <- tail(vals[order(vals[,2]),],drop_n_worst)[,1]
  print(paste('Drop',worst_n))
  .train <- .train %>% select(-all_of(worst_n))
  .test <- .test %>% select(-all_of(worst_n))
  
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
