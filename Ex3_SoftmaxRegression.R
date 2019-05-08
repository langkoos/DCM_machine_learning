#Ex.3: Mode Choice
#Libraries
library(nnet)
library(dplyr)
#Set working directory
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course")
#Load data
modedata <- read.csv("data/apollo_modeChoiceData.csv") 
#omit unnecessary columns
modedata<- modedata[,-which(names(modedata) %in% c('ID','RP','SP','RP_journey','SP_task'))]

#Steps
#Step 1) Check factor variables
#Step 2) Check continuous variables
#Step 3) Train/test set
#Step 4) Build the model
#Step 5) Assess the performance of the model


#Step 1) Check factor variables
# Select categorical column
factorlabels <- c('av_car','av_bus','av_air','av_rail','service_air','service_rail',
                  'female','business','choice')
modedata[,factorlabels]<-lapply(modedata[,factorlabels],factor)

#Step 2) Check continuous variables
continuous <-select_if(modedata, is.numeric)
summary(continuous)
#Standarize numeric columns
modedata <- modedata %>% mutate_if(is.numeric, funs(as.numeric(scale(.))))

#Step 3) Train/test set
N <-nrow(modedata) #total number of rows
percentage_training <- 0.7 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N))
test_index  <- (1:nrow(modedata))[-train_index]

#Step 4) Build the model
#Softmax Regression (aka Multinomial Logistic Regression)

#Declare reference level for Softmax Regression
modedata$choice <- relevel(modedata$choice, ref='1')

#Use multinom to fit model on train data
mlr <- multinom(choice~ ., data=modedata[train_index,])

#Step 5) Assess the performance of the model
#Predict on test data
y_hat<-predict(mlr,modedata[test_index,])

#Confusion matrix
mlr.matrix<-table(y_hat,modedata[test_index,'choice'])

#Accuracy
mlr.acc<- sum(diag(mlr.matrix))/sum(mlr.matrix)
mlr.acc


