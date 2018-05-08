
# coding: utf-8

# # Data Information:
# 
# 
# ## Input variables:
# 
# ### Background Data:
# 1 - bu : League or Tournament the match took place in (categorical)
# 
# 2 - year : Year the match took place in (numerical)
# 
# 3 - Season : Spring or Summer depending on which half of the year the match took place in  (categorical: 'Spring', 'Summer',)
# 
# 4 - Type:(categorical: 'Season', 'Playoffs',' Regional', or 'International match')
# 
# 5 - Address: website address the data is scraped from
# 
# ### Game Data(only some of them are shown here):
# 
# #### Every feature that starts with 'b' or 'blue' has its corresponding feature in 'r' or 'red'. 'blue' and 'red' refer to the two team in a single match
# 
# 6 - blueTeamTag: Blue Team's tag name (ex. Team SoloMid is TSM)
# 
# 7 - redTeamTag:	Red Team's Tag Name (ex. Team SoloMid is TSM)
# 
# 8 - gamelength:	Game length in minutes
# 
# 9 - golddiff:	Gold difference - computed Blue minus Red - by minute
# 
# 10 - goldblue:	Blue Team's total gold value by minute
# 
# 11 - bKills: List of Blue Team's kills - [Time in minutes, Victim, Killer, Assist1, Assist2, Assist3, Assist4, x_pos, y_pos]
# 
# 12 - bTowers:	List of minutes that Blue Team destroyed a tower and Tower Location
# 
# 13 - bInhibs:	List of minutes that Blue Team destroyed an inhibitor and Location
# 
# 14 - bDragons:	List of minutes that Blue Team killed a dragon Dragon Type
# 
# 15 - bBarons: List of minutes that Blue Team killed a baron nashor
# 
# 16 - bHeralds:	List of minutes that Blue Team killed a rift herald
# 
# 17 - goldred:	Red Team's total gold value by minute
# 
# 18 - blueTop:	Name of Blue Team's player in the top position
# 
# 19 - blueTopChamp:	Name of Blue Team's champion in the top position
# 
# 20 - goldblueTop:	Blue's Top position player's gold value by minute
# 
# 21 - blueBans:	List of champions Blue Team banned (in order)
# 
# ## Output variable (desired target):
# 
# 22 - y ( bResult ): Result of the match for Blue Team - 1 is a win, 0 is a loss

# # Data preprocessing 

# ### Import libraries and data
# 

# In[1]:
import warnings

warnings.filterwarnings('ignore')
    
from sklearn import datasets
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

lol_origin = pd.read_csv('./leagueoflegends/leagueoflegends.csv')
lol = lol_origin.copy()
lol.tail()


# ### Data Dropping and Data Extraction.
# 
# First, we need to **drop some data**. The reasons are as follow:
# 
# 1. Some data are **unrelated** to our goal, such as 'Address',which is only a website address.
# 
# 
# 2. Some data are **unavailable** at 15 minutes after the game begins, such as 'gamelength', which refers to the time span of the whole game. Also, 'bBarons(rBarons)' involves a neutral unit that appears in the game 20 minutes after it begins.
# 
# 
# 3. Some data are just **replaced**, such as 'goldblue', 'golddiff'. The former one is the total gold of the blue team, thus it is the sum of the gold in different position (i.e. Top, Middle, Jungle, ADC, Support). I drop it in order to advoid Perfect Collinearity. Similarly, 'golddiff' is just the difference of the gold of the two teams.
# 
# After that, if I treat 'player' as ordinal variables and transform them into dummies, I will also drop the original features 
# 
# that indicate their names.
# 
#  Also, data at 15 minutes are extracted from lists of minute data for some features.

# In[2]:

#lol = lol_origin.drop(['golddiff','goldblue','goldred','Address','bBarons','rBarons','redBans','blueBans','gamelength'],1) 
#this is the way we treat 'player' as nominal variables

lol = lol_origin.drop(['golddiff','goldblue','goldred','Address','bBarons','rBarons','redBans','blueBans','blueTop','blueMiddle','blueJungle','blueADC','blueSupport',
          'redTop','redMiddle','redJungle','redADC','redSupport','gamelength'],1) 
# this is the way we treat 'player' as ordinal variables, therefore, we set them as dummies later

for x in ['goldblueTop','goldblueJungle','goldblueMiddle','goldblueADC','goldblueSupport',
          'goldredTop','goldredJungle','goldredMiddle','goldredADC','goldredSupport']:
    lol[x] = [int(y.split(',')[14]) for y in lol_origin[x]]

for x in ['Top','Jungle','Middle','ADC','Support']:
    lol['golddiff' + x ] = lol['goldblue' + x ] - lol['goldred' + x ]
    del lol['goldblue' + x ]
    del lol['goldred' + x ]


# For more complicated cases such as 'bKills' below, a function was used to extract the accummulated data at 15 minutes.

# In[3]:

lol['bKills'].values[1]


# In[4]:

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


# In[5]:

for x in ['bKills','bTowers','bInhibs','bDragons','bHeralds',
          'rKills','rTowers','rInhibs','rDragons','rHeralds']:
    del lol[x]
    lol[x] = get_data15(lol_origin[x])


# Here, I collect all the champs' names, players' names and teams' names into three list for later use in LabelEncoder.
# 
# One **benefit** is that I can make sure that every champ( player or team) is mapped into the same number no matter what color it belongs to. ('red' or 'blue) 

# In[6]:

champ_names = lol_origin['blueTopChamp']
player_names = lol_origin['blueTop']
team_names = lol_origin['blueTeamTag']
for x in ['blueMiddleChamp','blueJungleChamp','blueADCChamp','blueSupportChamp',
          'redTopChamp','redMiddleChamp','redJungleChamp','redADCChamp','redSupportChamp']:
    champ_names = champ_names.append(lol_origin[x])
for x in ['blueMiddle','blueJungle','blueADC','blueSupport',
          'redTop','redMiddle','redJungle','redADC','redSupport']:
    player_names = player_names.append(lol_origin[x])
team_names = team_names.append(lol_origin['redTeamTag'])


# ### Processing the ordinal and nominal data,seperately

# In this part, **LabelEncoder** is used to transform nominal variables.
# 
# And **pd.get_dummies** is used to transform ordinal variables.
# 
# After **get_dummies**, the total features will be more than 3000. Given that the total samples are only around 8000, this 
# 
# seems to be a problem. But I also did some research, and it seems that dummies variable can be seen as, to some extent, 
# 
# certain variables ( Players and Champs in my case). If so, the fetures will be around 60 and the samples are enough.

# In[7]:

from sklearn.preprocessing import LabelEncoder

def nominal_transform(names,sets):
    le = LabelEncoder()
    le.fit_transform(names.astype(str))
    for x in sets:
        lol[x] = le.transform(lol_origin[x].astype(str))


# In[8]:

Position = ['blueTop','blueMiddle','blueJungle','blueADC','blueSupport',
          'redTop','redMiddle','redJungle','redADC','redSupport']
Champ = [x+'Champ' for x in Position]
Team = ['blueTeamTag','redTeamTag']
Other = ['bu','Season','Type']

nominal_transform(champ_names,Champ)
#nominal_transform(player_names,Position)   #this is the way we treat 'player' as nominal variables
nominal_transform(team_names,Team)

for x in Other:
    le = LabelEncoder()
    lol[x] = le.fit_transform(lol_origin[x].astype(str))


# In[9]:

dummies_Player = pd.get_dummies(lol_origin[['blueTop','blueMiddle','blueJungle','blueADC','blueSupport',
          'redTop','redMiddle','redJungle','redADC','redSupport']])
# transform players' names into dummies


# In[10]:

Champ_list = sorted(champ_names.unique())
lol_origin.redBans = lol_origin.redBans.str.strip('[]').str.replace("'",'')
lol_origin.blueBans = lol_origin.blueBans.str.strip('[]').str.replace("'",'')
dummies_Champ_Bans = pd.DataFrame(np.zeros((len(lol_origin),len(Champ_list))),columns=Champ_list)
for i, Champ in enumerate(lol_origin.redBans):
    Champ = str(Champ)
    for ban in Champ.split(', '):
        dummies_Champ_Bans.ix[i, ban] = 1
for i, Champ in enumerate(lol_origin.blueBans):
    Champ = str(Champ)
    for ban in Champ.split(', '):
        dummies_Champ_Bans.ix[i, ban] = 1
del dummies_Champ_Bans['']
lol = lol.join(dummies_Champ_Bans)
lol = lol.join(dummies_Player) # treat players as ordinal variables
#lol.to_excel('lol_players_nominal.xlsx')   # export nominal_player data into excel, ordinal data is too big to export.


# ### Define X,y and Check Sample Balance

# In[11]:

y = lol['bResult']
X = lol.drop(['bResult','rResult'],1)


# In[12]:

labels = 'BlueWin', 'RedWin'
sizes = [y.value_counts()[1],y.value_counts()[0]]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ### Splitting data into 70% training and 30% test data

# In[13]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# ### Bringing features onto the same scale(Standardization)

# In[14]:

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)


# # Feature Importance

# ## Random Forest Selection
# 

# In[15]:

from sklearn.ensemble import RandomForestClassifier

feat_labels = X.columns

forest = RandomForestClassifier(n_estimators=100, min_samples_split=3,min_samples_leaf =2, class_weight= {1:0.456, 0:0.544},
                                max_features = 'log2',n_jobs=8, criterion='gini',random_state = 1)

forest.fit(X_train, y_train)

print('accuracy of trainning = ' ,forest.score(X_train,y_train))
print('accuracy of testing = ' ,forest.score(X_test,y_test))

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(0,31):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plot_number = 31    
plt.title('Feature Importance')
plt.bar(range(0,plot_number), 
        importances[indices[0:plot_number]],
        align='center')

plt.xticks(range(0,plot_number), 
           feat_labels[indices[0:plot_number]], rotation=90)
plt.xlim([-1, plot_number])
plt.ylim([0.00000,0.06])
plt.savefig('/random_forest_selection.png', dpi=300)
plt.show()


# From the result we can see that the first and most important group of features are **gold difference** in different positions. 
# 
# And the second important group of features are **accumulated 'Kills','Towers' and 'Dragons'** that one team obtains at 15 minutes. 
# 
# And then **champs(heroes) that player use** form the third group of features. 
# 
# And finnaly, despite that most players do not account for the result of the game, **top player** actually help predict the game's outcome. ( Wolf was once regarded as the best support in the world, and Band was also top ADC in the world.) 

# ## PCA Selection
# 
# This is used to advoid overfitting and reduce computation.

# In[16]:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
pca.explained_variance_ratio_


# In[17]:

plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.savefig('PCA explained variance ratio.png')
plt.show()


# Here I choose only 100 PCA components because of the limited computational power of my computer.

# In[18]:

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# # Evaluation mertic
# 
# ## Confusion Matrix
# 

# In[19]:

import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Metric Output Function for Specific Models
# 
# Here I simply combine all the metric in one function, therefore the later codes will look more concise.

# In[20]:

def plot_MO(model,X_train,X_test):   
    class_names = ['Bluewin','Bluelose']
    
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print('accuracy of trainning = ' ,accuracy_score(y_train,y_pred_train))
    print('accuracy of testing = ' ,accuracy_score(y_test,y_pred))
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.savefig(str(model)[0:10]+'.png')
    plt.show()
    
    prs = precision_score(y_test, y_pred)
    rcs = recall_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred)
    
    print('PRS =',prs)
    print('RCS =',rcs)
    print('F1 score =',f1s)


# ## Hyperparameters Optimization
# 
# This is a function to find out the optimal hyperparameters. Again, defining a function is for more concise codes.
# 
# Moreover, X_train was defined as a input **because I use different data to train different model**, i.e. X_train_pca were 
# 
# used to train SVC model and X_train were used to train the others.
# 
# Finally, given that we are just predicting the result of a game and the samples are relatively balanced, I just choose 
# 
# **'recall'** as the scoring method.

# In[21]:

from sklearn.model_selection import GridSearchCV
def hyper_op(model,param_grids,X_train = X_train,cv=2):
    grid_search = GridSearchCV(model, param_grid=param_grids, scoring='recall', cv=cv)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


# # Training Data with Different Models
# 
# The standard procedure in this part is:
# 
# 1. Using **GridSearch** to find out best hyperparameters in a certain range
# 
# 2. Using the hyperparameters obtained from step 1 to train the model
# 
# 3. Using **plot_MO** we defined to assess the model

# ### Logistic Regression

# In[22]:

from sklearn.linear_model import LogisticRegression

param_grids = {'C':[0.01,0.1]}

lr2 = LogisticRegression(penalty='l1',C=0.1,random_state=1)
hyper_op(lr2,param_grids,X_train = X_train)


# In[23]:


lr2 = LogisticRegression(penalty='l1',C=0.1,random_state=1)
lr2 = lr2.fit(X_train, y_train)

plot_MO(lr2,X_train,X_test)


# 

# ### Support Vector Machine
# 
# **Note that** I set 'probability = True', because we need to construct a majority proba model later.

# In[27]:

from sklearn.svm import SVC

param_grids = {
              'C':[1,5]}

svm = SVC(kernel='linear', C=1, random_state=1,probability = True)
hyper_op(svm,param_grids,X_train = X_train_pca)


# In[32]:


svm = SVC(kernel='rbf', C=1, random_state=1,probability = True)
svm.fit(X_train_pca, y_train)

plot_MO(svm,X_train_pca,X_test_pca)


# ### XGboost

# In[26]:

from xgboost.sklearn import XGBClassifier
param_grids = {
    
    'learning_rate': [0.01, 0.02],
    'subsample':[0.8,1]
}
xgb_clf = XGBClassifier(learning_rate=0.01, n_estimators=50 , max_depth=6, min_child_weight=1,
                        subsample=1, colsample_bytree=0.7,
                        reg_alpha = 1, gamma=0.2,objective='binary:logistic',
                        scale_pos_weight=1.2, silent=0, seed=1000)

hyper_op(xgb_clf,param_grids,X_train = X_train)


# In[28]:

from xgboost.sklearn import XGBClassifier
xgb_clf = XGBClassifier(learning_rate=0.01, n_estimators=50 , max_depth=6, min_child_weight=1,
                        subsample=0.6, colsample_bytree=0.7,
                        reg_alpha = 1, gamma=0.2,objective='binary:logistic',
                        scale_pos_weight=1.2, silent=0, seed=1000)
xgb_clf.fit(X_train,y_train)

plot_MO(xgb_clf,X_train,X_test)


# ### Random Forest
# 
# **Note that** we have trained the model in 'feature importance' part. And the parameters used above are from here. It's a little bit tricky but basically I did this part first and then changed the parameters above.

# In[28]:

param_grids = {
    'n_estimators':[100,500],

}
forest = RandomForestClassifier(n_estimators=500, min_samples_split=5,min_samples_leaf =2, class_weight= {1:0.456, 0:0.544},
                                max_features = 'log2',n_jobs=8, criterion='gini',random_state = 1)
hyper_op(forest,param_grids,X_train = X_train)


# In[29]:
forest = RandomForestClassifier(n_estimators=1000, min_samples_split=3,min_samples_leaf =2, class_weight= {1:0.456, 0:0.544},
                                max_features = 'log2',n_jobs=8, criterion='gini',random_state = 1)
forest.fit(X_train,y_train)

plot_MO(forest,X_train,X_test)


# ## Majority proba
# 
# Since we train the models with different data, the embeded 'Majority Voting' method can not be applied directly.
# 
# Therefore we just calculate the weighted probability of different models' prediction to bulid a combined model.

# In[30]:

def stack_model(weight_lr = 0.25, weight_svm = 0.25,weight_forest = 0.25, plot=True):
    
    '''
    this part is unnecessary if we have these models trained before
    
    
    lr2 = LogisticRegression(penalty='l1',C=0.1,random_state=1)
    lr2 = lr.fit(X_train, y_train)
    
    svm = SVC(kernel='linear', C=1, random_state=1)
    svm.fit(X_train_pca, y_train)
    
    xgb_clf = XGBClassifier(learning_rate=0.1, n_estimators=50 , max_depth=4, min_child_weight=1,
                        subsample=0.65, colsample_bytree=0.7,
                        reg_alpha = 1, gamma=0.02,objective='binary:logistic',
                        scale_pos_weight=1.2, silent=0, seed=1000)
    xgb_clf.fit(X_train,y_train)
    
    forest = RandomForestClassifier(n_estimators=500, min_samples_split=5,min_samples_leaf =2, class_weight= {1:0.456, 0:0.544},
                                max_features = 'log2',n_jobs=8, criterion='gini',random_state = 1)

    forest.fit(X_train, y_train)
    
    
    '''
    prob_lr = lr2.predict_proba(X_test)[:,1]*weight_lr
    prob_svm = svm.predict_proba(X_test_pca)[:,1]*weight_svm
    prob_forest = forest.predict_proba(X_test)[:,1]*weight_forest
    prob_xgb = xgb_clf.predict_proba(X_test)[:,1]*(1-weight_lr-weight_forest-weight_svm)
    
    y_pred = np.c_[prob_lr, prob_svm,prob_forest, prob_xgb].sum(axis=1).round()
    
    if plot:
        
        class_names = ['Bluewin','Bluelose']

        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()
    
    f1s = f1_score(y_test,y_pred)
    rcs = recall_score(y_test, y_pred)
    prs = precision_score(y_test, y_pred)
   
    print('PRS =',prs),print('RCS =',rcs),print('F1 score =',f1s)
    
    return f1s, rcs, prs, prob_lr,prob_svm,prob_forest,prob_xgb 


# Similarly, we go through different weights to find out the best weights according to some scoring methods.

# In[33]:

f1s_list=[]
rcs_list=[]
prs_list=[]
for w1 in [0.05,0.1,0.2,0.25,0.3,0.4]:
    for w2 in [0.05,0.1,0.2]:
        for w3 in [0.05,0.1,0.2,0.25,0.3]:
            
            f1s_one, rcs_one, prs_one = stack_model(w1, w2, w3,plot = False)[0:3]
            f1s_list.append(f1s_one)
            rcs_list.append(rcs_one)
            prs_list.append(prs_one)
            # print(w1, w2, w3, f1s_one, rcs_one, prs_one)


# In[34]:

plt.plot(f1s_list,label='f1s')
plt.plot(rcs_list,label='rcs')
plt.plot(prs_list,label='prs')
plt.legend(loc='best')
plt.savefig('f1s with different weights.png')
plt.show()


# ## Evaluating (AUC)
# 
# Here I just plot the ROC_AUC for all the models.
# 
# The fact that combined model performs worse than Logistic Regression, Random Forest and XGboost probabily because of the bad 
# 
# performance of SVM model. And that could be the result of less dimensional data.

# In[40]:

prob_lr,prob_svm,prob_forest,prob_xgb  = stack_model(0.05,0.05,0.05,plot=False)[3:]
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

all_clf = [lr2, xgb_clf,forest]
label =['Logistic Regression','XGboost','Random Forest']

colors = ['black', 'blue','orange']
linestyles = [':', '-.','steps']
for clf, label, clr, ls         in zip(all_clf,
               label, colors, linestyles):

    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))
y_pred = svm.predict_proba(X_test_pca)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                 y_score=y_pred)
roc_auc = auc(x=fpr, y=tpr)
plt.plot(fpr, tpr,
         color='green',
         linestyle='--',
         label='%s (auc = %0.2f)' % ('SVM', roc_auc))

y_pred = np.c_[prob_lr, prob_svm, prob_xgb].sum(axis=1).round()
fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                 y_score=y_pred)
roc_auc = auc(x=fpr, y=tpr)
plt.plot(fpr, tpr,
         color='red',
         linestyle='-',
         label='%s (auc = %0.2f)' % ('Proba_Weighted', roc_auc))

    
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')


plt.savefig('ROC_AUC.png', dpi=300)
plt.show()


# 
