library(data.table)
library(xgboost)
train<-fread('train.csv')
campaign<-fread('campaign_data.csv')
test<-fread('test_BDIfz5B.csv')

trainds<-merge(train,campaign,by="campaign_id",all.x = T)
testds<-merge(test,campaign,by="campaign_id",all.x = T)

set.seed(888)


#############COMMON#############################

intersect<-fintersect(trainds[,'user_id',with=FALSE],testds[,'user_id',with=FALSE])
testintersect<-merge(testds,intersect,by="user_id")

click<-trainds[,list(meanclick=mean(is_click,na.rm=T)),by="user_id"]
trainds<-merge(trainds,click,by="user_id")
testintersect<-merge(testintersect,click,by="user_id")

open<-trainds[,list(meanopen=mean(is_open,na.rm=T)),by="user_id"]
trainds<-merge(trainds,open,by="user_id")
testintersect<-merge(testintersect,open,by="user_id")


trainds$ind<-1
testintersect$ind<-2


train_test<-rbind(trainds,testintersect,fill=T)

train_test[,":="(noofcampaign=length(unique(campaign_id))),by="user_id"]
train_test[,":="(noofusers=length(unique(user_id))),by="campaign_id"]

for (f in setdiff(names(train_test),'id')) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


traincommon<-subset(train_test,ind==1)
testcommon<-subset(train_test,ind==2)



features<-c('noofcampaign','meanclick','noofusers','total_links'
            ,'meanopen','communication_type')

xgb_params = list(
  eta = 0.05,
  objective = 'binary:logistic',
  eval_metric='auc',
  colsample_bytree=0.7,
  subsample=0.7,
  min_child_weight=30
)


dtrainmat = xgb.DMatrix(as.matrix(traincommon[,features,with=FALSE]), label=traincommon$is_click)
#cv<-xgb.cv(xgb_params, dtrainmat, nrounds =500,verbose= 1,num_threads=4
#           ,nfold = 5,early_stopping_rounds = 20)

#nround=which(cv$evaluation_log$test_auc_mean==max(cv$evaluation_log$test_auc_mean))

gbdt = xgb.train(xgb_params, dtrainmat, nrounds =100,verbose= 1,num_threads=4)

dtestmat = xgb.DMatrix(as.matrix(testcommon[,features,with=FALSE]))
predcommon = predict(gbdt,dtestmat)
summary(predcommon)
pred1<-((predcommon-min(predcommon))/(max(predcommon)-min(predcommon)))

dfcommon<-data.frame(id=testcommon$id,pred=pred1)

imp<-xgb.importance(feature_names = features,model=gbdt)
imp

######################UNIQUE###############################

train<-fread('train.csv')
campaign<-fread('campaign_data.csv')
test<-fread('test_BDIfz5B.csv')

trainds<-merge(train,campaign,by="campaign_id",all.x = T)
testds<-merge(test,campaign,by="campaign_id",all.x = T)

set.seed(888)



diff<-fsetdiff(testds[,'user_id',with=FALSE],trainds[,'user_id',with=FALSE])

testunique<-merge(testds,diff,by="user_id")

library(stringi)

trainds$wordcount<-stri_count(trainds$subject,regex="\\S+")
testds$wordcount<-stri_count(testds$subject,regex="\\S+")

trainds$ind<-1
testunique$ind<-2


train_test<-rbind(trainds,testunique,fill=T)

for (f in setdiff(names(train_test),'id')) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


trainds<-subset(train_test,ind==1)
testunique<-subset(train_test,ind==2)


features<-c('total_links','no_of_internal_links','no_of_images'
            ,'communication_type'
            #,'email_body'
            )

xgb_params = list(
  eta = 0.05,
  objective = 'binary:logistic',
  eval_metric='auc',
  colsample_bytree=0.7,
  subsample=0.7,
  min_child_weight=30
)


dtrainmat = xgb.DMatrix(as.matrix(trainds[,features,with=FALSE]), label=trainds$is_click)
#cv<-xgb.cv(xgb_params, dtrainmat, nrounds =500,verbose= 1,num_threads=4
#           ,nfold = 5,early_stopping_rounds = 20)

#nround=which(cv$evaluation_log$test_auc_mean==max(cv$evaluation_log$test_auc_mean))

gbdt = xgb.train(xgb_params, dtrainmat, nrounds =130,verbose= 1,num_threads=4)

dtestmat = xgb.DMatrix(as.matrix(testunique[,features,with=FALSE]))
predunique = predict(gbdt,dtestmat)
summary(predunique)
pred2<-((predunique-min(predunique))/(max(predunique)-min(predunique)))
summary(pred2)

dfunique<-data.frame(id=testunique$id,pred=pred2)

imp<-xgb.importance(feature_names = features,model=gbdt)
imp

totalpred<-rbind(dfcommon,dfunique)
df<-data.frame(id=totalpred$id,is_click=totalpred$pred)
#Create File for Submission
write.csv(df,'xgb_9_19_common_unique.csv',row.names = F,quote = F)

