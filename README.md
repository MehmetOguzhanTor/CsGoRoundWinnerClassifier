# CsGoRoundWinnerClassifier

Description of the Dataset
The name of the dataset that we are using for the project is CS:GO Round Winner Classification which is about a video game called Counter Strike:Global Offensive. CS:GO is a first person shooter game. There are two teams which are called Counter-Terrorists (CT) and Terrorists (T). The objective of the game is to win rounds either by eliminating the opposing team or capturing/protecting a specific area in the map.

The dataset is found on Kaggle, yet it was originally published by Skybox which was part of their Al challenge about the game. The Skybox included about 700 demos from several professional level tournaments that were played in 2019 and 2020. The restarts and warmups were not included in the demos. The recordings have been snapped every 20 second as the round continues. The number of snaps, which will be our instances, is 122411. The Terrorist team wins 62430 rounds and the Counter-Terrorist team wins 59981 of them [1]. So, the distribution of the instances is balanced and there are no missing values in the dataset. Therefore, we are expecting a good result while making the classification. The attribute number of the dataset is 97. These attributes consist of the time left in the current round, the current score of the Counter-Terrorist team, the map the round is being played on, if the bomb has been planted or not, the total bankroll of all Terrorist players, number of helmets on the Counter-Terrorist team etc.

Description of the Question
Aforementioned dataset will be used for answering the question “Which team is expected to win the round?”. This question will be answered by using different methods that we will learn throughout the semester.

Detailed Description of the Methods Used
As it was mentioned in the project proposal we are planning to use 3 different Machine Learning methods that will calculate and predict the possible winner of the CS:GO Round separately. The 3 methods will be Naive Bayes, Logistic Regression and Decision Tree. Firstly, Feature selection should be implemented to the dataset to exclude the features that are not needed or/and spoil some equations when they have all 0 values. In order to use the methods, the dataset should be separated into 2 groups which are Train and Test sets. We have decided to separate them as 80% of them used for training and 20% of them used for testing. In the process, the only libraries used for the project will be “numpy, pandas, etc.”. We will derive some mathematical formulas in order to calculate some probabilities. The project will be constructed without the usage of any extensive Python machine learning libraries.

Preprocessing and Simulation Setup
First of all, we are using Google Colaboratory for writing and executing our python code which is a well-suited environment for machine learning algorithms. As we started the coding, the first thing to do was importing the simple libraries like numpy and pandas. Then we read and save the dataset inside the Colab in order to work on it. When we analyzed the data deeply, the first thing we saw was some of the features never change as they are always 0 and have no contribution to the classification of the round winner label. We could observe the corresponding feature in the following snapshot.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/9c8ced99-9a23-4ffc-af71-e20f118e0e93)

Figure 1: The Features Eliminated with Feature Selection

Before starting any other modification, we have decided to eliminate these 6 features. It is because we realized that in the process of normalizing the data, since these 6 features have all 0 values, there will be 0 in the dividing part of the normalizing equation which will give runtime errors in the process. After the elimination there are 90 features and 1 label, for a total of 91 columns.

The second important thing is that the data is taken from snapshots that are taken every 20 seconds. So, in the dataset, there are visible continuations in the values for many features unless the round changes. In order to work on a fairly distributed train and test sets, we have decided to shuffle all of the rows in the dataset so that the algorithms will work better. However, before shuffling we needed to be sure all of the results in the matrix are numerical. In the dataset, there are also some strings that will make our jobs difficult. We decided to give numerical numbers to all of them by using a dictionary. For example, for the map feature there are 8 different possible results for which we gave numbers from 1 to 8. So, the string results are replaced with the numerical results in the dataset. Moreover, there is only one feature that has a Boolean result. Again, we replaced it with an integer number such that 1 for a True result and 0 for a False result. As it can be observed from the following snapshot, map names are now numerated and bomb planted has float values instead of Boolean values.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/5da2b3cc-4fdf-4f47-8e74-82bbf8262878)

Figure 2: Dataset with Corrected Attribute Instances

After the adjustment and shuffling of the dataset, in the next process for the beginning of the algorithm, we removed the label vector from the data to calculate and compare the predicted result with it. It is important that we remove the label vector after shuffling the data in order to know which label represents which row. Then, we needed to derive some parameters like mean or standard deviation, which will be useful for future calculations. In order to find them, we describe the transpose of the dataset. The result matrix is in Figure 4. Yet, description does not provide us with the normalized matrix, for this we use the following formula in our algorithm. At this point we have realized that it gives better results if we divide the train data into 2 parts which are CT winning data and T winning data and then separately take their normalizations and use these results to train our algorithm.

![Screenshot 2023-12-23 153018](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/2391fdb0-3a10-4880-a85a-03ef15fa6e90)

Eqn. 1: Normalization Formula

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/69d74848-0579-42c9-af09-b979389ea754)

Figure 3: Described Dataset

