setwd('C:\\Users\\rober\\Desktop\\RAND_pro\\prog_calc\\prog_calc2\\Data\\cle')

data = read.csv("Rand_train.csv", header=TRUE, sep=',')
data_val = read.csv("Rand_valid.csv", header=TRUE, sep=',')
data_test = read.csv("Rand_test.csv", header=TRUE, sep=',')

#Transform cost
data$zero <- ifelse(data$cost == 0, 0, 1)  # 0 if zero costs
data_test$zero <- ifelse(data_test$cost == 0, 0, 1)  # 0 if zero costs
data_val$zero <- ifelse(data_val$cost == 0, 0, 1)  # 0 if zero costs

v1 = 194 
v2 = 513
v3 = 901
v4 = 1403
v5 = 2330
v6 = 5672


### gamma twopart
h1.rmse <- glm(zero ~ . -cost, family=binomial(link=logit), data=data)
pred2 <- predict(h1.rmse, data_val, type="response")
pred3 <- predict(h1.rmse, data_test, type="response")

h2.rmse <- glm(cost ~ . -zero, family = Gamma(link = log), data=subset(data, zero == 1))

dataTrans = subset(data_val, zero == 1)
predsTrans = predict(h2.rmse, dataTrans, type="response")
preds3 <- pred3 * predict(h2.rmse, data_test, type="response")
mean(abs(preds3 - data_test$cost))
#329.5136

y1 = data_test$cost[data_test$cost<v1]
yh1 = preds3[data_test$cost<v1]
y2 = data_test$cost[data_test$cost>=v1 & data_test$cost<v2]
yh2 = preds3[data_test$cost>=v1 & data_test$cost<v2]
y3 = data_test$cost[data_test$cost>=v2 & data_test$cost<v3]
yh3 = preds3[data_test$cost>=v2 & data_test$cost<v3]
y4 = data_test$cost[data_test$cost>=v3 & data_test$cost<v4]
yh4 = preds3[data_test$cost>=v3 & data_test$cost<v4]
y5 = data_test$cost[data_test$cost>=v4 & data_test$cost<v5]
yh5 = preds3[data_test$cost>=v4 & data_test$cost<v5]
y6 = data_test$cost[data_test$cost>=v5 & data_test$cost<v6]
yh6 = preds3[data_test$cost>=v5 & data_test$cost<v6]
y7 = data_test$cost[data_test$cost>=v6]
yh7 = preds3[data_test$cost>=v6]
 
mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))
mean(abs(yh4 - y4))
mean(abs(yh5 - y5))
mean(abs(yh6 - y6))
mean(abs(yh7 - y7))



#Tweedie
library(cplm)
tw.rmse <- cpglm(cost ~., data=data)
preds5 <- predict(tw.rmse, data_test, type="response")
mean(abs(preds5 - data_test$cost))
#320.6379

y1 = data_test$cost[data_test$cost<v1]
yh1 = preds5[data_test$cost<v1]
y2 = data_test$cost[data_test$cost>=v1 & data_test$cost<v2]
yh2 = preds5[data_test$cost>=v1 & data_test$cost<v2]
y3 = data_test$cost[data_test$cost>=v2 & data_test$cost<v3]
yh3 = preds5[data_test$cost>=v2 & data_test$cost<v3]
y4 = data_test$cost[data_test$cost>=v3 & data_test$cost<v4]
yh4 = preds5[data_test$cost>=v3 & data_test$cost<v4]
y5 = data_test$cost[data_test$cost>=v4 & data_test$cost<v5]
yh5 = preds5[data_test$cost>=v4 & data_test$cost<v5]
y6 = data_test$cost[data_test$cost>=v5 & data_test$cost<v6]
yh6 = preds5[data_test$cost>=v5 & data_test$cost<v6]
y7 = data_test$cost[data_test$cost>=v6]
yh7 = preds5[data_test$cost>=v6]

mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))
mean(abs(yh4 - y4))
mean(abs(yh5 - y5))
mean(abs(yh6 - y6))
mean(abs(yh7 - y7))


### Tobit
library(VGAM)
tob.rmse <- vglm(cost ~ . -zero, tobit(Lower = 0, type.fitted = c("censored")), data = data, maxit=100)
preds <- predict(tob.rmse, data_test, type="response")
mean(abs(preds - data_test$cost))
#291.1773

y1 = data_test$cost[data_test$cost<v1]
yh1 = preds[data_test$cost<v1]
y2 = data_test$cost[data_test$cost>=v1 & data_test$cost<v2]
yh2 = preds[data_test$cost>=v1 & data_test$cost<v2]
y3 = data_test$cost[data_test$cost>=v2 & data_test$cost<v3]
yh3 = preds[data_test$cost>=v2 & data_test$cost<v3]
y4 = data_test$cost[data_test$cost>=v3 & data_test$cost<v4]
yh4 = preds[data_test$cost>=v3 & data_test$cost<v4]
y5 = data_test$cost[data_test$cost>=v4 & data_test$cost<v5]
yh5 = preds[data_test$cost>=v4 & data_test$cost<v5]
y6 = data_test$cost[data_test$cost>=v5 & data_test$cost<v6]
yh6 = preds[data_test$cost>=v5 & data_test$cost<v6]
y7 = data_test$cost[data_test$cost>=v6]
yh7 = preds[data_test$cost>=v6]

mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))
mean(abs(yh4 - y4))
mean(abs(yh5 - y5))
mean(abs(yh6 - y6))
mean(abs(yh7 - y7))


#baseline
preds0 = exp(data_test$lpcost) - 0.5
mean(abs(preds0 - data_test$cost))  
#344.2663  

y1 = data_test$cost[data_test$cost<v1]
yh1 = preds0[data_test$cost<v1]
y2 = data_test$cost[data_test$cost>=v1 & data_test$cost<v2]
yh2 = preds0[data_test$cost>=v1 & data_test$cost<v2]
y3 = data_test$cost[data_test$cost>=v2 & data_test$cost<v3]
yh3 = preds0[data_test$cost>=v2 & data_test$cost<v3]
y4 = data_test$cost[data_test$cost>=v3 & data_test$cost<v4]
yh4 = preds0[data_test$cost>=v3 & data_test$cost<v4]
y5 = data_test$cost[data_test$cost>=v4 & data_test$cost<v5]
yh5 = preds0[data_test$cost>=v4 & data_test$cost<v5]
y6 = data_test$cost[data_test$cost>=v5 & data_test$cost<v6]
yh6 = preds0[data_test$cost>=v5 & data_test$cost<v6]
y7 = data_test$cost[data_test$cost>=v6]
yh7 = preds0[data_test$cost>=v6]

mean(abs(yh1 - y1))
mean(abs(yh2 - y2))
mean(abs(yh3 - y3))
mean(abs(yh4 - y4))
mean(abs(yh5 - y5))
mean(abs(yh6 - y6))
mean(abs(yh7 - y7))

