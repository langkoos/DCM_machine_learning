library(dplyr)
library(magrittr)
library(ranger)
library(nnet)
library(SOMbrero)
library(tibble)
library(ggplot2)


accuracy <- function(df) {
  (diag(df) %>% sum) / sum(df)
}

swissmetro <-
  read.delim("~/git/DCM_course/swissmetro/pandas/swissmetro.dat")


sample_ids <-
  sample(1:nrow(swissmetro),
         size = nrow(swissmetro) * .5,
         replace = F)


# data conditioning -------------------------------------------------------


sm <-
  swissmetro %>%
  mutate(
    PURPOSE = factor(PURPOSE),
    FIRST = factor(FIRST),
    TICKET = factor(TICKET),
    WHO = factor(WHO),
    LUGGAGE = factor(LUGGAGE),
    AGE = factor(AGE),
    MALE = factor(MALE),
    INCOME = factor(INCOME),
    GA = factor(GA),
    ORIGIN = factor(ORIGIN),
    DEST = factor(DEST),
    TRAIN_AV = factor(TRAIN_AV),
    CAR_AV = factor(CAR_AV),
    SM_AV = factor(SM_AV),
    SM_SEATS = factor(SM_SEATS),
    CHOICE = factor(CHOICE)
  )

sm.train <-   
  sm %>%
  slice(sample_ids)

sm.test <-   
  sm %>%
  slice(-sample_ids)


# random forest -----------------------------------------------------------


rf <-
  sm.train %>%
  select(-1:-4, -SM_AV,-TRAIN_AV,-ORIGIN,-DEST) %>% 
  ranger(CHOICE ~ ., data = ., importance = "impurity")
rf

importance(rf) %>%
  tibble(var = names(.), val = .) %>%
  arrange(val %>% desc) %>%
  qplot(
    data = .,
    geom = 'col',
    x = var %>% reorder(-val),
    y = val
  )


rf.pred <- predict(rf, data = sm.test)

sm.test$rf <- rf.pred$predictions
sm.test %>% 
  select(CHOICE, rf) %>% 
  table() %>% 
  accuracy()



# ml estimation -----------------------------------------------------------


mlr <- 
  sm.train %>% 
  select(-1:-4, -SM_AV,-TRAIN_AV,-ORIGIN,-DEST) %>% 
  multinom(CHOICE ~ ., data = .)


sm.test$mlr <- predict(mlr, sm.test)

sm.test %>% 
  select(CHOICE, mlr) %>% 
  table() %>% 
  accuracy()


