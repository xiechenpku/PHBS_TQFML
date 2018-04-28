# Proposal 

## Introduction:

League of Legends (LoL) is an immensely popular multiplayer online battle arena game, with over 100 million monthly active users worldwide.
10 players are divided into 2 teams (blue or red). The objective for each team is to destroy the opposing teams' "Nexus", which can be thought as the main building in a base. Destroy the enemy Nexus and your team will win. The offical holds the matches in different regions, including the NALCS(for North American), LCK(for South Korean), LPL(for China), EULCS, LMS, and CBLoL leagues as well as the World Championship. Last year, World Championship was held in China and millions of fans were attracted. More than 100 million unique viewer watched the Final. So LoL is really influential in China.This project is designed to predict the battle result based on all the game information we have at 15 minutes(since the game begins).

## Data Source:

League of Legends competitive matches between 2015-2018. From Kaggle's website https://www.kaggle.com/chuckephron/leagueoflegends 

## Features:

golddiff: the economy difference between the two teams.

blueTop: name of blue team's player in TOP position.

blueTopChamp: name of blue team's Champ in TOP position.

goldblueTop: gold value of blue team's player.

...

## Methods:

Since this is a classification problem, KNN/logistic/SVM/RandomForest will be tried to solve the problem.

