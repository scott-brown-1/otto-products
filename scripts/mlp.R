#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(bonsai)

setwd('..')
source('./scripts/utils.R')
source('./scripts/feature_engineering.R')
PARALLEL <- F
FACTOR_CUTOFF <- 25

#########################
####### Load Data #######
#########################

## Load data
data <- load_data(factor_cutoff=0) #FACTOR_CUTOFF
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

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifer Model ##
#########################

## Define model
mlp_model <- mlp(
  hidden_units = 22,#tune(),
  epochs = 75, #or 100 or 2507
  activation="relu"
) %>%
  #set_engine("nnet", verbose=0) %>%
  set_engine("keras", verbose=0) %>%
  set_mode('classification')

## Define workflow
mlp_wf <- workflow() %>%
  add_recipe(prepped_recipe) %>%
  add_model(mlp_model)

# ## Grid of values to tune over
# tuning_grid <- grid_regular(
#   hidden_units(range=c(1, 30)),
#   levels = 5)
# 
# ## Split data for CV
# folds <- vfold_cv(train, v = 5, repeats=1)
# 
# ## Run the CV
# cv_results <- mlp_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# 
# ## Find optimal tuning params
# best_params <- cv_results %>%
#   select_best("accuracy")
# 
# print(best_params)
# 
# hidden_unit_plot <- cv_results %>% 
#   collect_metrics() %>%
#   filter(.metric=="accuracy") %>%
#   ggplot(aes(x=hidden_units, y=mean)) + 
#   geom_line()
# 
# ggsave('hidden_unit_plot.png',plot=hidden_unit_plot)

## Fit workflow
final_wf <- mlp_wf %>%
  #finalize_workflow(best_params) %>%
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
vroom::vroom_write(output,'./outputs/mlp_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}