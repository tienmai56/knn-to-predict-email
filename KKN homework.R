
####################################################################################################################

####################################################################################################################

####################################################################################################################

#############################This program was created by Davit Khachatryan##########################################
######©2016-2017 by Davit Khachatryan.  All rights reserved. No part of this document may  be reproduced or#########
######transmitted in any form or by any means, electronic, mechanical, photocopying, recording or otherwise#########
############################without prior written permission of Davit Khachatryan###################################

####################################################################################################################

####################################################################################################################

####################################################################################################################


library(gmodels)
library(class) #The package that supports KNN. You have to install it first before loading with 
#the "library" command.

#START OF DATA IMPORT

#update the path below to point to the directory and name of your data in *.csv format  

mydata=read.csv("C:/Users/tmai1/Documents/My RStudio/Rawdata/UniversalBank2019.csv")
str(mydata)

#END OF DATA IMPORT


#START OF VARIABLE REDEFINITION


mydata$myresponse=mydata$Personal.Loan #Substitute "RESPONSE" with the name of your response variable
mydata$Personal.Loan=NULL #Substitute "RESPONSE" with the name of your response variable

#Comment (i.e. put under a hashtag) the statement below ONLY IF the variable you are modeling is read into RStudio as
#a factor. Otherwise, leave it as un-commented and run it to convert it to a factor.

mydata$myresponse=as.factor(mydata$myresponse) 


str(data)

#END OF VARIABLE REDEFINITION


#The statements below remove all the variables that will not be passed to the knn algorithm
#as predictors. If no such redundant variables exist in your dataset, then the statements
#in the "REDUNDANT VARIABLE REMOVAL" section should be deleted or commented out.

#START OF REDUNDANT VARIABLE REMOVAL

mydata$ID=NULL #Substitute "OBS." with the name of the variable in your data that 
#will not be passed to the knn algorithm. Add as many statements similar 
#to this as needed.

mydata$ZIP.Code=NULL  #Substitute "CHK_ACCT" with the name of the variable in your data that 
#will not be passed to the knn algorithm.

mydata$Family=NULL
mydata$CCAvg=NULL
mydata$Education=NULL
mydata$Mortgage=NULL
mydata$Personal.Loan=NULL
mydata$Securities.Account=NULL
mydata$CD.Account=NULL
mydata$Online=NULL
mydata$CreditCard=NULL


#END OF REDUNDANT VARIABLE REMOVAL


#DONOT MODIFY THE NEXT TWO LINES OF CODE
str(mydata)
summary(mydata)
raw_for_export=mydata


str(mydata)
#Use the following statement to standardize the numeric predictors which are in need of
#standardization. You need to carefully list the numbers of the columns that need to be 
#standardized.Be careful NOT to standardize columns that are categorical or the column corresponding to "myresponse".
#NOTE: If none of the predictors need to be standardized, then the entire code from
#'START OF VARIABLE STANDARDIZATION' to 'END OF VARIABLE STANDARDIZATION' need to be either
#deleted or commented (i.e. put under hashtags)

#START OF VARIABLE STANDARDIZATION

col_nums=c(1,2,3); #Substitute (1,2,3) with the possitions at which columns that
#are in need of standardization appear in the updated "mydata"
#dataframe. You can find out the numbers of columns by running
#the str(mydata) command.In this example, columns listed 
#as the 1st, 2nd and 3rd will be standardized.
#If you know that the columns in need of standardization are 
#located starting from column #1 and ending at column #N
#Then you can replace the 'c(1,2,3)' with 'seq(1,N)' for more efficient coding

#############################################################################################
#####################DO NOT MODIFY THE LINES BELOW UNTIL WHERE IT SAYS#######################
#############################"END OF VARIABLE STANDARDIZATION"###############################


cols_for_standard=as.matrix(mydata[,col_nums])
standardized=scale(cols_for_standard)
all_col_nums=c(1:length(names(mydata)))
remaining_cols=as.vector(all_col_nums[is.na(pmatch(all_col_nums, col_nums))])
remaining_data=subset(mydata,select=remaining_cols)
mydata=cbind(remaining_data, standardized)

#END OF VARIABLE STANDARDIZATION


#############################################################################################
#####################################ATTENTION###############################################
#############################################################################################

#######################IF THE ABOVE MODIFICATIONS ARE MADE CORRECTLY,########################
####AT THIS POINT "MYDATA" DATA FRAME SHOULD CONTAIN ONLY THE PREDICTORS AND THE OUTCOME.#### 
####IN CASE IT CONTAINS ANYTHING MORE OR LESS, THE CODE BELOW WILL NOT FUNCTION PROPERLY.####
#############################################################################################


str(mydata)

#############################################################################################
########################DO NOT MODIFY LINES BELOW UNTIL WHERE IT SAYS########################
#######################################"END KNN"#############################################


#START DATA BREAKDOWN FOR HOLDOUT METHOD

nobs=dim(mydata)[1]
set.seed(1) #sets the seed for random sampling

prop = prop.table(table(mydata$myresponse))
prop
length.vector = round(0.8*nobs*prop)
train_size=sum(length.vector)
test_size=nobs-train_size
class.names = as.data.frame(prop)[,1]
numb.class = length(class.names)
train_index = c()

for(i in 1:numb.class){
  index_temp = which(mydata$myresponse==class.names[i])
  train_index_temp = sample(index_temp, length.vector[i], replace = F)
  train_index = c(train_index, train_index_temp)
}


train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
test=mydata[-train_index,]#everything not in the training set should go into testing set

y_train=train$myresponse
y_test=as.data.frame(test$myresponse)


pred_train=train
pred_test=test
pred_train$myresponse=NULL
pred_test$myresponse=NULL


dim(pred_train) #confirms that training data has only 80% of observations
dim(pred_test) #confirms that testing data has 20% of observations

#END DATA BREAKDOWN FOR HOLDOUT METHOD

#START KNN

#Note, that below I am capping the maximum value of K to be 20.
#More specifically, if the number of rows in training data are less than 200 then 
#the maximum possible value for K is going to be equal to the 10% of the dimension
#of the training data. Otherwise, the highest value it is going to be 20. 

rate=matrix(0,100,min(round(dim(train)[1]/10),20))#initialize
percent_correct=c(1:min(round(dim(train)[1]/10),20))#initialize

#Note, that since in "knn" function the ties are broken at random,
#running the function multiple times may result in slightly different results
#for each value of K. For that reason, for each K I am running the function 100
#times and averageing the results.

for (i in 1:100){
  for (j in 1:min(round(dim(train)[1]/10),20)){
    nearest=knn(train=pred_train, test=pred_test, cl=y_train, k=j)
    rate[i,j]=100*sum(nearest==as.character(y_test[,1]))/dim(y_test)[1]
  }
}


for (i in 1:length(percent_correct)){
  percent_correct[i]=mean(rate[,i])
}

percent_correct

#END KNN

#############################################################################################
############################SPECIFICATION OF THE FINAL KNN###################################
#############################################################################################

#You will need to choose the value for K that resulted in the highest pecentage of correct
#classifications for the test data set, based on "percent_correct" list. That value of
#K needs to be subsequently passed to function KNN below to display the best classification
#That value of K needs to be substituted for "15" in the last argument in the code below.

#START FINAL KNN RUN FOR THE BEST VALUE OF K

nearest_final=knn(train=pred_train, test=pred_test, cl=y_train, k=9)

#############################################################################################
############################DO NOT MODIFY BEYOND THIS POINT##################################
#############################################################################################


percent_correct=100*sum(as.character(nearest_final)==as.character(y_test[,1]))/dim(y_test)[1]

#END FINAL KNN RUN FOR THE BEST VALUE OF K

#The code below calculates the performance of a simple classificaiton, which just assigns 
#each observation from the test set to the class that is dominating the training set.
#We then compare the performance of the KNN with best K (as found above) to this simple
#classification, to find out how KNN compares to this "naive" benchmark.

#START BENCHMARKING COMPARISON. Confirming that the simple classification
#rule yields a correct classification rate equal to the perentage of the 
#dominant class in the entire population.

prop_train = as.data.frame(prop.table(table(train$myresponse)))
prop_train=prop_train[order(-prop_train$Freq),]
dominant_class=prop_train[1,1]
test_benchmark=test
test_benchmark$simple_classification=as.character(dominant_class)
percent_correct_simple=100*sum(test_benchmark$simple_classification==as.character(y_test[,1]))/dim(y_test)[1]

#END BENCHMARKING COMPARISON

#START: PRINTING OUT SOME METRICS OF ACCURACY  
print(paste("Percentage of Correct Classifications for 'Best' KNN is:",percent_correct, "percent")) 
print(paste("Percentage of Correct Classifications for the Benchmark Classification is:",percent_correct_simple, "percent"))             

colnames(y_test)="myresponse"
table_for_export=cbind(raw_for_export[-train_index,],nearest_final)
table_for_export$knn_classification=table_for_export$nearest_final
table_for_export$nearest_final=NULL


#Getting the confusion matrix below  
Confusion_Matrix = CrossTable(table_for_export$myresponse, table_for_export$knn_classification,dnn=c("True Class","Predicted Class"), prop.chisq=F,prop.t=F, prop.c=F, prop.r=F)

#END: PRINTING OUT SOME METRICS OF ACCURACY   

#####################################################################################################  
#If you have truly new records for which you do not know their true classifications,
#but would like to predict using KNN with the "best" value of K found above, you will need to 
#make sure that those records are stored in a data frame called "fresh", and ensure that the mentioned
#data frame has all the predictors that were in your testing set and nothing more. The names of the predictors
#should be exactly the same as the names of the predictors in "mydata".
#Data frame "fresh" needs to have 1 less column than the testing set (since there is a column called "myresponse" in your testing 
#set which won't be present in "fresh"). If the above noted specifications hold, then to generate the classifications
#for the records contained in "fresh" all you have to do is to consider the code below. 
#The resultant classifications will be stored in the last column in data frame called "table_with_classifications".
#####################################################################################################  

#Copy your fresh data into "backup"

#backup=fresh

#If you standardized the predictors earlier, then you have to standardize "fresh" before classifying.
#To do that, get rid of the hashtags from where it says "Standardize Fresh" to "End of Fresh Standardization",
#highlight the corresponding lines, and run. 
#Otherwise, if standardization was not carried out earlier, then proceed directly to where it says "Classifying Records in Fresh"

#Standardize Fresh  

#copy=raw_for_export[,1:(dim(raw_for_export)[2]-1)]
#mean.vec=apply(copy,2, mean)
#sd.vec=apply(copy,2, sd)
#fresh=fresh[,colnames(copy)]
#fresh=scale(fresh, center=mean.vec, scale=sd.vec)

#End of Fresh Standardization

#Classifying Records in Fresh  
#fresh_classifications=knn(train=mydata[,2:dim(mydata)[2]], test=fresh, cl=mydata$myresponse, k=which(percent_correct==max(percent_correct)))
#table_with_classifications=cbind(backup, fresh_classifications)
#############################################################################################
##############################THIS IS THE END OF THE MACRO###################################
#############################################################################################


