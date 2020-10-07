

# SP500 = read.csv("/Users/martha/Desktop/SP500 dec2004-dec2018.csv")
# SPprices = SP500$Close
# 
# min(SPprices)
# max(SPprices)
# 
# max(SPprices) - min(SPprices)

df = read.csv("/Users/martha/Desktop/Original Stathis Data.csv")

df$date = as.Date(df$date)
df = df[order(df$date),]


YrVec = substr(df$date,1,4)
DateIndex = min(which(YrVec=='2004'))
df1 = df[(DateIndex):nrow(df),]


xindex = c( which(names(df1)=='WN1.Comdty'),
            which(names(df1)=='QC1.Index'),
            which(names(df1)=='XB1.Comdty'),
            which(names(df1)=='ST1.Index') )
df2 = df1[-c(1,2),-xindex]




df1$date[which(df1$date=='2005-06-30')-313]

N = ncol(df2)
k = nrow(df2)
#dftest = matrix(0,nrow=k,ncol=N-1)
#dftest[1,] = df2[1,-1]
dftest = df2
for(tt in 2:N){
  datatest = dftest[,tt]
  for(i in 2:k){
    if(is.na(datatest[i])){
      dftest[i,tt] = datatest[i-1]
    }else{
      dftest[i,tt] = datatest[i]
    }
  }
}

dftest = df2
for(tt in 2:N){
  for(i in 2:k){
    if(is.na(dftest[i,tt])==TRUE){
      dftest[i,tt] = dftest[i-1,tt]
    }
  }
}


# REMOVED 2 COMMODITIES AND 2 INDICES BECAUSE TOOOOOO MANY NA VALUES!
write.csv(dftest,'/Users/martha/Desktop/Final Stathis Data (3).csv')





which(dftest$date=='2005-06-30') # = 389 and 388 on python
dftest$date[which(dftest$date=='2005-06-30')]








NAlist <- list()
NAlengthVec <- c()
for(tt in 2:ncol(df1)){
  x = df1[,tt]
  NAlist[[tt-1]] = which(is.na(x)==TRUE)
  NAlengthVec[tt-1] = length(NAlist[[tt-1]])
}

y = which(NAlengthVec>130) + 1

df2 = df1[-y,]

NAlist1 <- list()
NAlengthVec1 <- c()
for(tt in 2:ncol(df2)){
  x = df2[,tt]
  NAlist1[[tt-1]] = which(is.na(x)==TRUE)
  NAlengthVec1[tt-1] = length(NAlist1[[tt-1]])
}




z = Reduce(union,NAlist1)
df3 = df2[-z,]

NAlist2 <- list()
NAlengthVec2 <- c()
for(tt in 2:ncol(df3)){
  x = df3[,tt]
  NAlist2[[tt-1]] = which(is.na(x)==TRUE)
  NAlengthVec2[tt-1] = length(NAlist2[[tt-1]])
}

DateCount <- c()
for(j in 2:nrow(df3)){
  DateCount[j-1] = df3$date[j] - df3$date[j-1]
}
unique(DateCount)


write.csv(df3,'/Users/martha/Desktop/Final Stathis Data (2).csv')

datIndex = 


which(df3$date=='2011-01-04')
df3$date[3648]



ncol(df3)

std_vec <- c()
N = nrow(df3)
for( tt in 2:ncol(df3)){
  returns = (df3[2:N,tt] - df3[1:(N-1),tt])/df3[1:(N-1),tt]
  std_vec[tt-1] = std(returns) 
}
std_vec*sqrt(252)




minvec <- c()
maxvec <- c()
for( tt in 2:ncol(df3)){
  minvec[tt-1] = min(df3[,tt])
  maxvec[tt-1] = max(df3[,tt]) 
}
minmaxvec = cbind(minvec,maxvec)
minmaxvec
diffvec = minmaxvec[,2] - minmaxvec[,1]
minmaxvec = cbind(minmaxvec,diffvec)

index = which(minmaxvec[,3]>2000)+1

df4 = df3[,-c(index)]




minvec1 <- c()
maxvec1 <- c()
for( tt in 2:ncol(df4)){
  minvec1[tt-1] = min(df4[,tt])
  maxvec1[tt-1] = max(df4[,tt]) 
}
minmaxvec1 = cbind(minvec1,maxvec1)
diffvec1 = minmaxvec1[,2] - minmaxvec1[,1]
minmaxvec1 = cbind(minmaxvec1,diffvec1)
minmaxvec1


write.csv(df4,'/Users/martha/Desktop/Final Stathis Data (1).csv')



install.packages("roll")
library(roll)

N = ncol(df4)
K = nrow(df4)
ReturnsMat <- matrix(NA,K-1,N-1)
std_vec <- c()
for(tt in 2:N){
  #ReturnsMat[,tt-1] = (df4[2:K,tt] - df4[1:(K-1),tt])/df4[1:(K-1),tt]
  ReturnsMat[,tt-1] = df4[2:K,tt] - df4[1:(K-1),tt]
  std_vec[tt-1] = std(ReturnsMat[,tt-1])*sqrt(252)
}

sigma_target = mean(std_vec)
sigma_target


action_vec = df2tensor([MACD_signal(data,t-1),MACD_signal(data,t-2)])

r_vec = data[1:(t+1)] - data[0:t]
r_df  = pd.DataFrame(r_vec)
ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

sigma_vec = df2tensor([ex_ante_sigma[-2],ex_ante_sigma[-3]])


roll_sd(SPreturns,60)

N = length(SPprices)
SPreturns1 = (SPprices[2:N] - SPprices[1:(N-1)])/SPprices[1:(N-1)]

std(SPreturns1[1:312])
mean(SPreturns1[1:312])

std(SPreturns1)
x = roll_sd(SPreturns1,60)
x[length(x)]


tt = 3406
TT = 3648

ExampleData = df3$FV1.Comdty[1:TT]
r_vec = data[2:tt] - data[1:(tt-1)]
#r_vec = (data[2:tt] - data[1:(tt-1)])/data[1:(tt-1)]
#r_vec = log(data[2:tt]/data[1:(tt-1)])
mean(r_vec)
std(r_vec)

std(movavg(x=r_vec,n=60,type='e'))


ewmsd <- function(x, alpha) {
  n <- length(x)
  sapply(
    1:n,
    function(i, x, alpha) {
      y <- x[1:i]
      m <- length(y)
      weights <- (1 - alpha)^((m - 1):0)
      ewma <- sum(weights * y) / sum(weights)
      bias <- sum(weights)^2 / (sum(weights)^2 - sum(weights^2))
      ewmsd <- sqrt(bias * sum(weights * (y - ewma)^2) / sum(weights))
    },
    x = x,
    alpha = alpha
  )
}

N = 60
alpha = (N-1)/(N)
alpha
x = ewmsd(r_vec,alpha)
n = length(x)
x[n]









