#Ex.2: Route Choice (Logistic Regression)
#Implement Gradient Descent for Logistic Regression and compare results against glm package 

#Generate fake data
#Note: y is 0 or 1
set.seed(2019)
X.mat <- cbind(rep(1,100),runif(100),runif(100))
y<-sample(c(0,1),100,replace = TRUE)
y<-as.numeric(X.mat[,2]*X.mat[,3]< mean(X.mat[,2]*X.mat[,3]))
plot(X.mat[,2],X.mat[,3],col=as.factor(y))

#initial b
betas<-rnorm(3)

#Sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}
sigmoid(X.mat%*%betas)

#Cost function (aka Loss Function)
cost <- function(beta, X, y){
  m <- length(y) # number of training examples
  
  h <- sigmoid(X%*%beta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}
cost(betas,X.mat,y)

#gradient function
grad <- function(beta, X, y){
  m <- length(y) 
  
  h <- sigmoid(X%*%beta)
  grad <- (t(X)%*%(h - y))/m
  grad
}
grad(betas,X.mat,y)

#Gradient Descent
logGradDescent=function(X,y,beta_init,learning_rate,n_iterations){
  beta=beta_init
  loss_history=rep(0,n_iterations)
  for(k in 1:n_iterations){
    beta=beta-learning_rate * grad(beta,X,y)
    loss_history[k]=cost(beta,X,y)
  }
  return(list(beta=beta,loss_history=loss_history))
}

#Test Gradient Descent
beta_init<-rnorm(3)
learning_rate<-0.5
n_iterations<-5500

results <- logGradDescent(X.mat,y,beta_init,learning_rate,n_iterations)

results$beta

plot(results$loss_history,pch=20,type='b',main = 'Gradient Descent for Logistic Regression')


#Compare with glm package ############################
gendata<-data.frame(cbind(y,X.mat[,2:3]))
names(gendata)<-c('y','X1','X2')
#y to factor
gendata$y<-as.factor(gendata$y)

#Logistic regression
lr <- glm(y~.,data=gendata,family = binomial)

lr$coefficients
results$beta
#Are they different results?

#Plot decision boundaries ##########################
#Our results
m<- -results$beta[2]/results$beta[3]
b<- -results$beta[1]/results$beta[3]
plot(X.mat[,2],X.mat[,3],col=as.factor(y))
abline(b,m,col='blue')

#R package results
m2<- -lr$coefficients[2]/lr$coefficients[3]
b2<- -lr$coefficients[1]/lr$coefficients[3]
abline(b2,m2,col='green')


