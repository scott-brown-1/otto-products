#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)

source('./scripts/utils.R')
source('./scripts/wrangling.R')
PARALLEL <- T

#########################
####### Load Data #######
#########################

## Load data
train <- load_train()
test <- load_test()

#########################
## Feature Engineering ##
#########################

set.seed(2003)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode=F, smote_K=5, pca_threshold=0.85)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifier Model #
#########################

## Define model
rf_model <- rand_forest(
  mtry = 12, #tune(),
  min_n = 40, #tune(),
  trees = 1000
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Define workflow
rf_wf <- workflow(prepped_recipe) %>%
  add_model(rf_model)

## Grid of values to tune over
# tuning_grid <- grid_regular(
#   mtry(range=c(1,5)),#(range=c(4,ncol(train))),
#   min_n(),
#   levels = 5)
# 
# ## Split data for CV
# folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
#cv_results <- rand_forest_wf %>%
#  tune_grid(resamples=folds,
#            grid=tuning_grid,
#            metrics=metric_set(roc_auc))

## Find optimal tuning params
#best_params <- cv_results %>%
#  select_best("roc_auc")

#tryCatch(
#  expr = {
#    print(best_params)
#  },
#  error = function(e){ 
#    print('Error caught')
#    print(e)
#  })

## Fit workflow
#final_wf <- rand_forest_wf %>%
#  finalize_workflow(best_params) %>%
#  fit(data = train)

final_wf <- rf_wf %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='prob') %>%
  bind_cols(test$id,.) %>%
  rename(
    id=...1,
    Class_1=.pred_Class_1,
    Class_2=.pred_Class_2,
    Class_3=.pred_Class_3,
    Class_4=.pred_Class_4,
    Class_5=.pred_Class_5,
    Class_6=.pred_Class_6,
    Class_7=.pred_Class_7,
    Class_8=.pred_Class_8,
    Class_9=.pred_Class_9)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/rf_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
