setwd('C:\\Users\\rober\\Desktop\\RAND_pro\\prog_calc\\prog_calc2\\Data\\cle')

data = read.csv("Rand_train.csv", header=TRUE, sep=',')
data_val = read.csv("Rand_valid.csv", header=TRUE, sep=',')
data_test = read.csv("Rand_test.csv", header=TRUE, sep=',')

#Transform cost
data$zero <- ifelse(data$cost == 0, 0, 1)  # 0 if zero costs
data_test$zero <- ifelse(data_test$cost == 0, 0, 1)  # 0 if zero costs
data_val$zero <- ifelse(data_val$cost == 0, 0, 1)  # 0 if zero costs

## gamma twopart
h1.rmse <- glm(zero ~ . -cost, family=binomial(link=logit), data=data)
pred2 <- predict(h1.rmse, data_val, type="response")
pred3 <- predict(h1.rmse, data_test, type="response")

h2.rmse <- glm(cost ~ . -zero, family = Gamma(link = log), data=subset(data, zero == 1))

dataTrans = subset(data_val, zero == 1)
predsTrans = predict(h2.rmse, dataTrans, type="response")
preds3 <- pred3 * predict(h2.rmse, data_test, type="response")
mean(abs(preds3 - data_test$cost))
#329.5136


cut0 = 0
cut1 = 1146

y1 = data_test$cost[data_test$cost==cut0]
yh1 = preds3[data_test$cost==cut0]
y2 = data_test$cost[data_test$cost>cut0 & data_test$cost<=cut1]
yh2 = preds3[data_test$cost>cut0 & data_test$cost<=cut1]
y3 = data_test$cost[data_test$cost>cut1]
yh3 = preds3[data_test$cost>cut1]

 
mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))


## gamma twopart (mode)
preds4 <- (pred3>0.5) * predict(h2.rmse, data_test, type="response")
mean(abs(preds4 - data_test$cost))
#350.8995


y1 = data_test$cost[data_test$cost==cut0]
yh1 = preds4[data_test$cost==cut0]
y2 = data_test$cost[data_test$cost>cut0 & data_test$cost<=cut1]
yh2 = preds4[data_test$cost>cut0 & data_test$cost<=cut1]
y3 = data_test$cost[data_test$cost>cut1]
yh3 = preds4[data_test$cost>cut1]


mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))





#Tweedie
library(cplm)
tw.rmse <- cpglm(cost ~., data=data)
preds5 <- predict(tw.rmse, data_test, type="response")
mean(abs(preds5 - data_test$cost))
#320.6379

y1 = data_test$cost[data_test$cost==cut0]
yh1 = preds5[data_test$cost==cut0]
y2 = data_test$cost[data_test$cost>cut0 & data_test$cost<=cut1]
yh2 = preds5[data_test$cost>cut0 & data_test$cost<=cut1]
y3 = data_test$cost[data_test$cost>cut1]
yh3 = preds5[data_test$cost>cut1]


mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))

### Tobit
library(VGAM)
tob.rmse <- vglm(cost ~ . -zero, tobit(Lower = 0, type.fitted = c("censored")), data = data, maxit=100)
preds <- predict(tob.rmse, data_test, type="response")
mean(abs(preds - data_test$cost))
#291.1773

y1 = data_test$cost[data_test$cost==cut0]
yh1 = preds[data_test$cost==cut0]
y2 = data_test$cost[data_test$cost>cut0 & data_test$cost<=cut1]
yh2 = preds[data_test$cost>cut0 & data_test$cost<=cut1]
y3 = data_test$cost[data_test$cost>cut1]
yh3 = preds[data_test$cost>cut1]


mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))


#baseline
preds0 = exp(data_test$lpcost) - 0.5
mean(abs(preds0 - data_test$cost))  
#344.2663  

y1 = data_test$cost[data_test$cost==cut0]
yh1 = preds0[data_test$cost==cut0]
y2 = data_test$cost[data_test$cost>cut0 & data_test$cost<=cut1]
yh2 = preds0[data_test$cost>cut0 & data_test$cost<=cut1]
y3 = data_test$cost[data_test$cost>cut1]
yh3 = preds0[data_test$cost>cut1]


mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))


#Q-Q plots

plot(sort(data_test$cost),sort(preds3), main="TPM")
abline(0, 1) 


plot(sort(data_test$cost),sort(preds5), main="Tweedie")
abline(0, 1) 


plot(sort(data_test$cost),sort(preds), main="Tobit")
abline(0, 1) 


plot(sort(data_test$cost),sort(preds0), main="Baseline")
abline(0, 1) 




