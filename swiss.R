library(dplyr)
library(magrittr)
library(ranger)
library(nnet)



swissmetro <-
  read.delim("~/git/DCM_course/swissmetro/pandas/swissmetro.dat")


sample_ids <-
  sample(1:nrow(swissmetro),
         size = nrow(swissmetro) / 2,
         replace = F)

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

sm.train <-   sm %>%
  select(-1:-4, -SM_AV, -TRAIN_AV) %>%
  slice(sample_ids)

sm.test <-   sm %>%
  select(-1:-4,-SM_AV, -TRAIN_AV) %>%
  slice(-sample_ids)

rf <-
  sm.train %>%
  ranger(CHOICE ~ ., data = .)
rf

rf.pred <- predict(rf, data = sm.test)

df <-
  table(sm.test$CHOICE, rf.pred$predictions)
df

acc <- (diag(df) %>% sum) / sum(df)
acc

sm.train.ml <- 
  sm.train %>% 
  mutate(
    CHOICE = relevel(CHOICE,ref=1)
  )

mlr <- multinom(CHOICE ~ ., data=sm.train.ml)


mlr.pred <- predict(mlr, data = sm.test)

df.mlr <-
  table(sm.test$CHOICE, mlr.pred)
df.mlr

acc.mlr <- (diag(df.mlr) %>% sum) / sum(df.mlr)
acc.mlr


  