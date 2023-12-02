2214049#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(bonsai)

setwd('..')
source('./scripts/utils.R')
source('./scripts/feature_engineering.R')
PARALLEL <- T
FACTOR_CUTOFF <- 25

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
  cl <- makePSOCKcluster(10)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode=F, poly=F, smote_K=0, pca_threshold=0)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifier Model #
#########################

boost_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = 0.1,#tune(),
  mtry = tune(),
  min_n = 2, #tune(),
  loss_reduction = 0 #tune(),
  ) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
boost_wf <- workflow(prepped_recipe) %>%
  add_model(boost_model)

# ## Grid of values to tune over
tuning_grid <- grid_regular(
  trees(),
  tree_depth(),
#   learn_rate(),
  mtry(range=c(3,20)),
#   min_n(),
#   loss_reduction(),
  levels = 4)

## Split data for CV
folds <- vfold_cv(train, v = 4, repeats=1)

# Run the CV
cv_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mn_log_loss))

# Find optimal tuning params
best_params <- cv_results %>%
  select_best("mn_log_loss")

print(best_params)

# Fit workflow
final_wf <- boost_wf %>%
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

vroom::vroom_write(output,'./outputs/light_gbm_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
