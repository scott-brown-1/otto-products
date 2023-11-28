#########################
### Imports and setup ###
#########################

library(DataExplorer)
library(ggmosaic)

source('./scripts/utils.R')

## Load data
train <- vroom::vroom('./data/train.csv')
test <- vrom::vroom('./data/test.csv')

###########################
####### Examine Data ######
###########################

## Exmaine dataframe and check data types; check shape
glimpse(train)
View(train)

## Check categorical vs discrete vs continuous factors
plot_intro(train)

###########################
### Check Missing Values ##
###########################

# View missing values by feature
plot_missing(train)

# Count total missing values
sum(sum(is.na(train)))
sum(sum(is.na(test)))

###########################
## Visually examine data ##
###########################

# ggplot(amazon, aes(x=ROLE_FAMILY), stat=count) +
#   geom_bar()

## Check for 0/1 feature columns
for(col in colnames(train)){
  if(length(unique(train[col])) == 2){
    print('A')
  }
} 

var_lengths <- c()
for(i in 1:dim(train)[2]){
  if(i==1 | i==95) next # Skip id and target cols
  var_lengths[i] <- nrow(unique(train[,i]))
} 

hist(var_lengths, breaks=100)

## NOTE: plot histogram of variable lengths

## NOTE: zero variance cols exist

###########################
### Examine response var ## 
###########################

head(train$target)

ggplot(train, aes(x=target), stat=count) +
  geom_bar()
