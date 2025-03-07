


# Load the library ####
rm(list=ls())
install.packages("mclust")
require(AUC)
library(AUC)# install.packages("AUC")
library(cvTools)# install.packages("cvTools")
library(caret)# install.packages("caret")
library("CrossValidate")# install.packages("CrossValidate")
library("doParallel")# install.packages("doParallel")
library(doParallel)# install.packages("doParallel")
library(doMC)# install.packages("doMC")
library(devtools)# install.packages("devtools")
library(DMwR) # install.packages("DMwR")
library("foreach")# install.packages("foreach")
library(pROC)# install.packages("pROC")
library(PRROC)# install.packages("PRROC")
library(plotly)# install.packages("plotly")
library(ROCR)# install.packages("ROCR")
library(rminer)# install.packages("rminer")
library(rpart)# install.packages("rpart")
library(randomForest)# install.packages("randomForest")
library(R.utils)# install.packages("R.utils")
library(R.matlab)# install.packages("R.matlab")
library(ROSE)# install.packages("ROSE")
library(iterators)# install.packages("iterators")
library(elasticnet)# install.packages("elasticnet")
library(e1071)# install.packages("e1071")
library(glmnet)# install.packages("glmnet")
library(ggplot2)# install.packages("ggplot2")
library(grid)# install.packages("grid")
library(lattice)# install.packages("lattice")
library(MASS)# install.packages("MASS")
library(ModelMetrics)# install.packages("ModelMetrics")
library(mclust)# install.packages("mclust")
library(nnet)# install.packages("nnet")
library(neuralnet) # install.packages("neuralnet")
library(ModelMetrics) # install.packages("ModelMetrics")
library(glmnet)
library(caret)
library(doParallel)


#### run in parallel ####
parallelnumber<-50
myCluster <- makeCluster(parallelnumber)
registerDoMC(parallelnumber)

# 5 fold nested cross validation
OuterKfold<-5
InnerKfold<-5
InnerIterNumber<-5

# create empty matrices
{
  multiResultClass <- function(result1=NULL){me <- list(
    result1 = result1)
  class(me) <- append(class(me),"multiResultClass")
  return(me)}
  
  predict_demo_all_outer<-NULL;predict_all_connect_all_outer<-NULL;predict_all_ensemble_all_outer<-NULL
  varImp_demo_all_outer<-NULL;varImp_SC_FC_all_outer<-NULL;varImp_connect_all_outer<-NULL
  r2_rmse_outer_demo_outerloop<-NULL; r2_rmse_outer_connect_outerloop<-NULL; r2_rmse_ensemble_all_outer<-NULL
  best_hyper_demo_outer<-NULL; best_hyper_connect_outer<-NULL; best_hyper_ensemble_outer<-NULL
  best_threshold_demo_all_outer<-NULL;best_threshold_all_connect_all_outer<-NULL;best_threshold_all_ensemble_all_outer<-NULL
  result_ConfMatrix_outer_demo<-NULL;result_ConfMatrix_outer_connect_all_outer<-NULL;result_ConfMatrix_outer_ensemble_all_outer<-NULL
  err_cv_outer<-NULL;varImp<-list();varimpnumber<-1
  dim_data_outer<-NULL
  varImp_haufe<-list()
  best_hyper_outer<-NULL

}

# start the model
Ridge_regression_results<-foreach(it=rep(1:50,2),# run 100 iterations
.combine = rbind,.multicombine=TRUE,.packages=c("nnet","caret","AUC","e1071","randomForest")) %dopar% {
cat(paste("iter=",it))

for(model in 1:length(data_used1)){

    cat(paste("model=",model))
      data_used<-data_used1[[model]]
      data_used<-as.data.frame(data_used)
      data_used$Gender<-as.factor(data_used$Gender)
      data_used$Dx<-as.factor( data_used$Dx)
      

      names1<-names(data_used)
      names(data_used)<-make.names(names1, unique = TRUE, allow_ = TRUE)
      data_used[,1]<-as.numeric(data_used[,1]) # make sure the outcome is numerical
      folds_outerloop<-createFolds(data_used[,1],k=OuterKfold,list = FALSE)  

for(outerloop in 1:OuterKfold){

  # create train and test datasets
  trainData <- data_used[folds_outerloop != outerloop, ]
  testData <- data_used[folds_outerloop == outerloop, ]
  
  err_cv<-NULL
  for (iterinner in 1:InnerIterNumber) {
    
    folds_outerloop_inner<-createFolds(trainData[,1],k=InnerKfold,list = FALSE)
    
    for(innerloop in 1:InnerKfold){
      
      trainData[,1]<-as.numeric(trainData[,1])
      
      # create training and validation datasets
      
      trainingData <- trainData[folds_outerloop_inner != innerloop, ]
      validationData <- trainData[folds_outerloop_inner == innerloop, ]
      
      # create interval for lambda
           lambdas<-10^seq(2, -3, by = -.5)
           
            for (lambda_connect in lambdas) {
              # training data
            trainingData<-na.omit(trainingData)
            innerx<-data.matrix(trainingData[,-1])
            innery<-as.matrix(trainingData[,1])
            # fit the model with training
            innermodel_enet_prob <- glmnet(innerx,innery,lambda=lambda_connect,alpha=0,family = "gaussian",standardize = FALSE) 
            
            # validation data
            validationData[,1]<-NULL
            # test the model with validation
            predict_validation_enet_prob<-predict(innermodel_enet_prob,newx = data.matrix(validationData),s=lambda_connect,type="response") 
            
            # evaluate the prediction accuracy using the validation data
            validationData <- trainData[folds_outerloop_inner == innerloop, ]
            preds <- predict_validation_enet_prob
            actual <- validationData[,1]
            preds<-as.numeric(preds)
            actual<-as.numeric(actual)
            rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
            tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
            rsq_innerloop <- 1 - rss/tss
            correlation<-cor.test(preds,actual,method="spearman")$estimate
            err_cv<-rbind(err_cv,c(iterinner,innerloop,lambda_connect,rsq_innerloop,correlation))

        }
  
      

    
  } # End inner loop ####
    
  
  } # end iterinner
  
  # Find the Best hyperparam ###
    param_median_auc<-NULL
    for(lambdabest in levels(as.factor(err_cv[,3]))){
      row1<-which(err_cv[,3]==lambdabest)
      param_median_auc<-rbind(param_median_auc,c(as.numeric(lambdabest),
                                                 as.numeric(median(err_cv[row1,5]))))
    }
    
    best_hyper<-param_median_auc[which.max(param_median_auc[,2]),1]
    
# recreate the train and test
    data_used[,1]<-as.numeric(data_used[,1])
    trainData <- data_used[folds_outerloop != outerloop, ]
    testData <- data_used[folds_outerloop == outerloop, ]
    
# scale the data
    normParam_train <- preProcess(trainData,method = c("center", "scale"))
    trainData <- predict(normParam_train, trainData)
    testData <- predict(normParam_train, testData)
  
    outerx<-data.matrix(trainData[,-1])
    outery<-data.matrix(trainData[,1])
    
# fit the model with the best hyperparameter and train dataset
    outermodel_enet_prob <- glmnet(outerx,outery,lambda=best_hyper,alpha=0,family = "gaussian",standardize = FALSE) 
    
    # variable importance
    varImp[[varimpnumber]]<-coef(outermodel_enet_prob,s=best_hyper)
    
    # HAUFE transformed variable importance ####
    parameter_coef<-outermodel_enet_prob$beta[,1];length(parameter_coef)
    predict_haufe_transform<-predict(outermodel_enet_prob,newx = data.matrix(outerx),s=best_hyper,type="response")

    covx<-cov(outerx)
    trainData <- data_used[folds_outerloop != outerloop, ]
    outery<-data.matrix(trainData[,1])
    haufe_transformed_beta_elvisha<-(covx%*% haufe_transformed_beta)/(sd(outery)^2)

    
    varImp_haufe[[varimplistnumber]]<-cbind(haufe_transformed_beta_elvisha,
                                            as.matrix(coef(outermodel_enet_prob,s=best_hyper)[-1]),
                                            rep(model,length(haufe_transformed_beta[,1])),
                                            rep(outerloop,length(haufe_transformed_beta[,1])),
                                            rep(it,length(haufe_transformed_beta[,1]))
    )

    varimplistnumber<-varimplistnumber+1
    varimpnumber<-varimpnumber+1
    
  # Predict the validation dataset
    testData[,1]<-NULL
  predict_test_enet_prob<-predict(outermodel_enet_prob,newx = data.matrix(testData),type="response") 

  testData <- data_used[folds_outerloop == outerloop, ]
  testData <- predict(normParam_train, testData)
  
  preds_test <- predict_test_enet_prob
  actual_test <- testData[,1]
  preds_test<-as.numeric(preds_test)
  actual_test<-as.numeric(actual_test)
  
  # calculate the accuracy metrics 
  rss <- sum((preds_test - actual_test) ^ 2)  ## residual sum of squares
  tss <- sum((actual_test - mean(actual_test)) ^ 2)  ## total sum of squares
  rsq_outerloop <- 1 - rss/tss
  rmse_outerloop<-rmse(actual_test, preds_test)
  
  # save prediction and actual outcomes
  predict_observed<-rbind(predict_observed,cbind(preds_test,
                                                 actual_test,
                                                 rep(model,length(actual_test)),
                                                 rep(outerloop,length(actual_test))
                                                 # rep(it,length(actual_test))
                                                 ))
  pred_actual[[outerloop]]<-cbind(preds_test,actual_test)
  correlation_outer<-cor.test(preds_test,actual_test,method="spearman")$estimate
  correlation_outer_pearson<-cor.test(preds_test,actual_test,method="pearson")$estimate
  
  # save the accuracy metrics
  r2_rmse_outer_outerloop<-rbind(r2_rmse_outer_outerloop,c(rsq=rsq_outerloop,
                                                           rmse=rmse_outerloop,
                                                           r_spearman=correlation_outer,
                                                           model=model,
                                                           r_pearson=correlation_outer_pearson,
                                                           outerloop=outerloop
                                                           ))


  
  } # end the outerloop
} # end the model loop

# save all results
result <- multiResultClass()
result$result1 <- r2_rmse_outer_outerloop
result$result2 <- dim_data_outer
result$result3 <- err_cv_outer
result$result4 <- cbind(testData[,1],predict_test_enet_prob)
result$result5 <- lambdas
result$result6 <- varImp_haufe
result$result7 <- varImp
result$result8 <- predict_observed
return(result)

}
