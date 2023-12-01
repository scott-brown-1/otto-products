#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(bonsai)
library(discrim)
library(stacks)

#setwd('..')
source('./scripts/utils.R')
source('./scripts/feature_engineering.R')
PARALLEL <- F
FACTOR_CUTOFF <- 25

#########################
####### Load Data #######
#########################

## Load data
data <- load_data(factor_cutoff=0)#FACTOR_CUTOFF)
train <- data$train
test <- data$test

#########################
## Feature Engineering ##
#########################

set.seed(2003)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(6)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode=F, poly=F, smote_K=0, pca_threshold=0)
bayes_recipe <- setup_train_recipe(train, encode=F, poly=F, smote_K=0, pca_threshold=0.85)
# TODO: Encode! Also test smote

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Create a control grid
untuned_model <- control_stack_grid()

#########################
##### BASE LIGHTGBM #####
#########################

boost_model <- boost_tree(
  trees = 75, 
  tree_depth = 2, 
  learn_rate = 0.1,
  mtry = tune(),#4,
  min_n = 2, 
  loss_reduction = 0 
  ) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

## Define workflow
boost_wf <- workflow(prepped_recipe) %>%
  add_model(boost_model)

## Grid of values to tune over
boost_grid <- grid_regular(
  mtry(range=c(4,6)),
  levels = 2
)

## Perform parameter tuning
tuned_boost_models <- boost_wf %>%
  tune_grid(
    resamples=folds,
    grid=boost_grid,
    metrics = metric_set(mn_log_loss),
    control = untuned_model)

#########################
#### BASE NAIVE BAYES ###
#########################

## Define model
bayes_model <- naive_Bayes(
  Laplace=0,
  smoothness=tune() #0.75
  ) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

## Define workflow
bayes_wf <- workflow() %>%
  add_recipe(bayes_recipe) %>%
  add_model(bayes_model)

## Grid of values to tune over
bayes_grid <- grid_regular(
  smoothness(),
  levels = 2)

## Perform parameter tuning
tuned_bayes_models <- bayes_wf %>%
  tune_grid(
    resamples=folds,
    grid=bayes_grid,
    metrics = metric_set(mn_log_loss),
    control = untuned_model)

#########################
# Stack models together #
#########################

# Create meta learner
model_stack <- stacks() %>%
  add_candidates(tuned_boost_models) %>%
  add_candidates(tuned_bayes_models)

# Fit meta learner
fitted_stack <- model_stack %>%
  blend_predictions() %>% # This is a Lasso (L1) penalized reg model
  fit_members()

## Predict new y
output <- predict(fitted_stack, new_data=test, type='prob') %>%
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

vroom::vroom_write(output,'./outputs/stacked_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
