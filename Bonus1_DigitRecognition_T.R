#Bonus 2: Handwritten Digit Recognition

set.seed(2019)

#Load dataset
setwd("/Users/cuauhtemoc/Documents/NUS Courses/ST4240 Data Mining/Assignments")
digits_train <- read.csv('Assignment 1/mnist83_train.csv')
digits_test <- read.csv('Assignment 1/mnist83_test.csv')

#Train data has 5000 examples, Test data has 3414 examples. Test data also includes labels.
ncol(digits_train)
ncol(digits_test)
nrow(digits_train)
nrow(digits_test)
#number of pixels = 784 = 28x28

#Plot two examples
par(mfrow = c(1,2))
im = matrix(as.numeric(digits_train[1,2:ncol(digits_train)]), nrow = 28, ncol= 28) #convert list to matrix for displaying
image(im/255, axes=FALSE, col=grey(seq(0,1,length=256)))

im = matrix(as.numeric(digits_train[3,2:ncol(digits_train)]), nrow = 28, ncol= 28) #convert list to matrix for displaying
image(im/255, axes=FALSE, col=grey(seq(0,1,length=256)))


########################################################################################################################################
#1. Logistic Regression (Lasso and Ridge Regularization) ##############################################################################

library(glmnet)

X_train = model.matrix(label ~ .,digits_train)
X_test = model.matrix(label ~ .,digits_test)
y_train = as.factor(digits_train[,'label'])   #for classification
y_test = as.factor(digits_test[,'label'])   #for classification

#CV Lasso Logistic Regression
lambda.grid = 10^seq(1,-5, length=100)  #lambda spectrum
cvfit.lasso = cv.glmnet(X_train, y_train, lambda=lambda.grid, alpha=1, family= 'binomial') #alpha=1 means Lasso, #family=binomial
cvfit.ridge = cv.glmnet(X_train, y_train, lambda=lambda.grid, alpha=0, family= 'binomial') #alpha=0 means Ridge, #family=binomial

par(mfrow=c(1,2))
plot(cvfit.ridge, main = 'Ridge CV')
plot(cvfit.lasso, main = 'Lasso CV')

#prediction
prediction.lasso = predict(cvfit.lasso, newx = X_test, s='lambda.min', type= 'class')
prediction.ridge = predict(cvfit.ridge, newx = X_test, s='lambda.min', type= 'class')
#Type is response because we would like a probability output. Class would give only the class

#par(mfrow=c(1,2))
#hist(prediction.ridge,nclass=100) 
#hist(prediction.lasso,nclass=100) #one can see that most predicted probabilities are close to 0% or 100%
#this may mean that it is very easy to make the difference between a male of a female voice

ridge.matrix<-table(prediction.ridge,y_test)
ridge.acc<- sum(diag(ridge.matrix))/sum(ridge.matrix)
ridge.acc

lasso.matrix<-table(prediction.lasso,y_test)
lasso.acc<- sum(diag(lasso.matrix))/sum(lasso.matrix)
lasso.acc

prediction.mlr = predict(cvfit.ridge, newx = X_test, s=0, type= 'class')
mlr.matrix<-table(prediction.mlr,y_test)
mlr.acc<- sum(diag(mlr.matrix))/sum(mlr.matrix)
mlr.acc


########################################################################################################################################
#2. Tree Methods (Random Forest) 

#label as factor
digits_train$label <- factor(digits_train$label)
digits_test$label <- factor(digits_test$label)


library(randomForest)

#Train random forest
rf <- randomForest(label ~ ., digits_train,ntree=100,mtry=floor(sqrt(785)))

#Predict on test data
rf.predict<-predict(rf,digits_test, type='class')

#Confusion matrix
rf.matrix<-table(rf.predict,digits_test[,'label'])

#Accuracy
rf.acc<- sum(diag(rf.matrix))/sum(rf.matrix)

rf.acc


########################################################################################################################################
#3. Neural Networks (Multilayer Perceptron)

#install.packages("devtools")
#require(devtools)
#install_github("rstudio/reticulate")
#install_github("rstudio/tensorflow")
#install_github("rstudio/keras")

#library(tensorflow)
#install_tensorflow(version = "nightly")

#devtools::install_github("rstudio/keras")
library(keras)
#install_keras()

###
##data input prep
y_train <- to_categorical(as.numeric(y_train)-1,num_classes = 2)
y_test <- to_categorical(as.numeric(y_test)-1, 2)

##model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(785)) %>% 
  #layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  #layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  #layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 12, activation = 'relu') %>%
  #layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  X_train, y_train, 
  epochs = 200, batch_size = 128, 
  validation_split = 0.2
)

#Evaluate accuracy
evaluate(model, X_test, y_test, verbose = 0)

