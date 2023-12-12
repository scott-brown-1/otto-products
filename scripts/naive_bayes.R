#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(discrim)

setwd('..')
source('./scripts/utils.R')
source('./scripts/feature_engineering.R')
PARALLEL <- F
FACTOR_CUTOFF <- 26

#########################
####### Load Data #######
#########################

## Load data
data <- load_data(factor_cutoff=FACTOR_CUTOFF)
train <- data$train
test <- data$test

#########################
## Feature Engineering ##
#########################

set.seed(2003)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(8)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode=T, poly=F, smote_K=5, pca_threshold=0.85)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifier Model #
#########################

## Define model
bayes_model <- naive_Bayes(
  Laplace=0,
  smoothness=tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

## Define workflow
bayes_wf <- workflow() %>%
  add_recipe(prepped_recipe) %>%
  add_model(bayes_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  smoothness(),
  levels = 4)

## Split data for CV
folds <- vfold_cv(train, v = 4, repeats=1)

## Run the CV
cv_results <- bayes_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mn_log_loss))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("mn_log_loss")

print(best_params)

## Fit workflow
final_wf <- bayes_wf %>%
  finalize_workflow(best_params) %>%
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

vroom::vroom_write(output,'./outputs/naive_bayes_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
