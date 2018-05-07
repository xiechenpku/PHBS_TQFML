Report
=
Introduction
-
The project is about the predication of the winner in a professional League of Legend game. The data are from Kaggle. The propasal can be seen [here](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/Readme.md) 

Data prepocessing
-
Let's first have a look at the raw data

![lol_origin](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/raw%20data.jpg)

The data are special because it contains time series data within a single sample.

For example, Let's look at **lol['bKills'].values[1]**, which means the kill that blue team obtains.

![lol_bKills](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/kill%20data.png)

And there are many features have the same structure. In order to extract these cummulated variables at 15mins, we define a function. Roughly speaking, the function goes through all individual kill and check if the time is higher than 15.

        def get_data15(feature):
                info = [x[2:-2] for x in feature]
                data15 = []
                for y in info:
                    y = y.split('], [')
                    j = 0
                    for z in y:
                        if z.strip() =='':
                            break
                        else:
                            z=z.split(',')
                            n =float(z[0])
                            if n < 15.0 :
                                j = j + 1
                    data15.append(j)
                return data15

After that, we need to deal with nominal and ordinal features. In my opinion, there are talented and normal players. But we can not rank heroes(Champs). Given that Faker, the world's top Middle lane player, choose a hero, I do not think I can beat him whatever hero I choose. Therefore, we treat player as ordinal variables by applying **get_dummies**. Meanwhile, Champs were transformed into nominal numbers by using **LabelEncoder**

And here is one drawback: Since we do not have any infomation about **new** heroes and players, we can not use the model when **new palyers** join the league or **new Champs** were created. 

The clean data look like this:

![clean_data](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/clean_data.png)

samples are relatively balanced:

![pie](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/pie.png)

Feature Selection
-

### Feature Importance ( through Random Forest)

![feature_importance](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/feature_importance.png)

I plot 30 most important features. we can see that the most important features are goldifference(economy), and then are enemies killed or tower destroyed, and then are champs. And finally, individual player help predict the result,too. All of these are quiet intuitive.

For example, South Korea is considered the best in League of Legend, and both Wolf and Band are top South Korean player.

### PCA

![PCA](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/PCA%20explained%20variance%20ratio.png)

I also use PCA to reduce dimensionality for later use in SVM model.

Apply Different Models
-
Before applying any model, we used **GridSearchCV** to find out the optimal hyper parameters in some range. 

### Logistic Regression

Accuracy of trainning = 0.8547056617922759

Accuracy of testing = 0.7305336832895888 

PRS = 0.7456964006259781 

RCS = 0.7660771704180064 

F1 score = 0.7557494052339413

![LR](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/lr2.png)

### SVM

SVM requires lots of computation, so for simplicity, I train SVM model with the data that have been dimensional reduced by PCA method.

accuracy of trainning =  0.8453318335208099

accuracy of testing =  0.6719160104986877

PRS = 0.672486033519553

RCS = 0.7741157556270096

F1 score = 0.7197309417040358

![SVM](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/SVC.png)

### Random Forest

accuracy of trainning =  0.775403074615673

accuracy of testing =  0.7235345581802275

PRS = 0.7554858934169278

RCS = 0.77491961414791

F1 score = 0.7650793650793651

![RF](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/RandomFore.png)


### XGboost


accuracy of trainning =  0.775403074615673

accuracy of testing =  0.7235345581802275

PRS = 0.7116182572614108

RCS = 0.8271704180064309

F1 score = 0.7650557620817844


![xg](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/XGBClassif.png)

### Proba Weighted Models

Since we use different data to train different models due to the limited computational power, the 'Majority Voting' in the book can not be applied. Thus, we add up the weighted average probability of different models through **.predict_proba()** method.

And the results of scores with different weights on different models look like this.

![SOCRES](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/f1s%20with%20different%20weights.png)

The F1 score is decreasing as the weight on XGboost model reduces.
Since F1 Score is the average of recall and precision, I choosed the weights that have the highest F1 score.

The ROC_AUC result

![roc](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/images/ROC_AUC.png)


