Report
=
Introduction
-
The project is about the predication of the winner in a professional League of Legend game. The data are from Kaggle. The propasal can be seen [here](https://github.com/xiechenpku/PHBS_TQFML/blob/master/Project/Readme.md) 

Data prepocessing
-
Let's first have a look at the raw data

![lol_origin](https://www.zybuluo.com/static/img/my_head.jpg)

The data are special because it contains time series data within a single sample.

For example, Let's look at **lol['bKills'].values[1]**

![lol_bKills]()

And there are many features have the same situation. In order to extract these cummulated variables at 15mins, we define a function.

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

After that, we need to deal with nominal and ordinal features. In my opinion, there are talented and normal pro players. But we can not rank heroes(Champs). Given that Faker, the world's top Middle lane player, choose a hero, I do not think I can beat him whatever hero I choose. Therefore, we treat player as ordinal variables by applying **get_dummies**. Meanwhile, Champs were transformed into nominal numbers by using **LabelEncoder**

And here is one drawback: Since we do not have any infomation about **new** heroes and players, we can not use the model when **new palyers** join the league or **new Champs** were created. 

the clean data look like this:

![clean_data]()

samples are relatively balanced:

![pie]

Feature Selection
-
### Feature Importance ( through Random Forest)

![feature_importance]()

### PCA

![PCA]()

Metric Results for Different Models
-
### Logistic Regression

### SVM

### Random Forest

### XGboost

### Proba Weighted Models
