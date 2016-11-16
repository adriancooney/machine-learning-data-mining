# http://blog.datacamp.com/machine-learning-in-r/

library(class)
library(gmodels)

illness <- read.csv("../illness-mapped.csv")

normalize <- function(x) {
    num <- x - min(x)
    denom <- max(x) - min(x)
    return (num/denom)
}

illness_norm <- as.data.frame(lapply(illness[1:8], normalize))

# Print out a summary of the data
summary(illness_norm)

set.seed(100)

# Create the sample
sampler <- sample(2, nrow(illness_norm), replace=TRUE, prob=c(0.67, 0.33))

illness_norm.train <- illness_norm[sampler==1, -1]
illness_norm.test <- illness_norm[sampler==2, -1]

illness_norm.cl <- illness[sampler==1, 9]
illness_norm.tl <- illness[sampler==2, 9]
k <- 3

illness_predicted <- knn(train = illness_norm.train, test = illness_norm.test, cl = illness_norm.cl, k)

CrossTable(x = illness_norm.tl, y = illness_predicted, prop.chisq=FALSE)

