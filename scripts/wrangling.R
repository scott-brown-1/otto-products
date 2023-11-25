library(embed) # for target encoding
library(themis) # for smote

setup_train_recipe <- function(df,encode=T,smote_K=5, pca_threshold=0.85){
  
  prelim_ft_eng <- recipe(target~., data=df) %>%
    step_zv(all_predictors()) # Remove zero-variance cols 

  if(encode){
    prelim_ft_eng <- prelim_ft_eng %>%
      step_lencode_mixed(all_nominal_predictors(), outcome = vars(target))
  }
  
  prelim_ft_eng <- prelim_ft_eng %>%
    step_normalize(all_numeric_predictors()) # Normalize features
  
  ## SMOTE upsample if K nearest neighbors > 0
  if(smote_K > 0){
    prelim_ft_eng <- prelim_ft_eng %>%
      step_smote(all_outcomes(), neighbors = smote_K)
  }
    
  ## Dimension reduce with principal component analysis if pca_threshold > 0
  if(pca_threshold > 0){
    prelim_ft_eng <- prelim_ft_eng %>%
      step_pca(all_predictors(), threshold=pca_threshold)
  }
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng, new_data=df)
  
  return(prepped_recipe)
}