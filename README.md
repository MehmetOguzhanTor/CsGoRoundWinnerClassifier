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

In the end of the preprocessing part, we derived all of the matrices, vectors and parameters for the following part. So, we have decided for train and test data and determine them with their corresponding label vector which has the round winner information. Moreover, we have normalized both train and test data by creating and using a normalization function that we created. After normalizing them separately, we observed that sometimes in the created normalization matrix, there are some ‘nan’ values which means normalization could not be done right. In the light of this observation, we realized that when we shuffle and separate the dataset into two parts, sometimes these train and/or test datasets have all 0 columns. The result of the situation is that some of the weapons are taken rarely in the snapshot. Therefore, when shuffled and separated, this situation could occur which harms our algorithm. So, we added an additional delete function to search for these ‘nan’ values and eliminate their attribute at that moment in order to continue with the process successfully.

When finding the mean and standard deviation of the set that we are working on, we used functions from the numpy library like np.mean and np.std which were really useful. When some of the resulting vectors were not correctly shaped after the adjustments, we reshaped them in order not to face any calculation error while executing the mathematical formulas. All of these preprocessing parts will be essential for the following Machine Learning algorithms. For this phase, they will be used for round winner classification with Naive Bayes classification.

Naive Bayes Classification
The firstly implemented classification method for the project is the Naive Bayes Method. In order to have a better classification result both the train data and test data were normalized beforehand. These normalized data were used in the Naive Bayes Method. For this method Gaussian probability distribution was utilized.

Gaussian Probability Distribution:
In the end of the preprocessing part, we derived all of the matrices, vectors and parameters for the following part. So, we have decided for train and test data and determine them with their corresponding label vector which has the round winner information. Moreover, we have normalized both train and test data by creating and using a normalization function that we created. After normalizing them separately, we observed that sometimes in the created normalization matrix, there are some ‘nan’ values which means normalization could not be done right. In the light of this observation, we realized that when we shuffle and separate the dataset into two parts, sometimes these train and/or test datasets have all 0 columns. The result of the situation is that some of the weapons are taken rarely in the snapshot. Therefore, when shuffled and separated, this situation could occur which harms our algorithm. So, we added an additional delete function to search for these ‘nan’ values and eliminate their attribute at that moment in order to continue with the process successfully.

When finding the mean and standard deviation of the set that we are working on, we used functions from the numpy library like np.mean and np.std which were really useful. When some of the resulting vectors were not correctly shaped after the adjustments, we reshaped them in order not to face any calculation error while executing the mathematical formulas. All of these preprocessing parts will be essential for the following Machine Learning algorithms. For this phase, they will be used for round winner classification with Naive Bayes classification.

Naive Bayes Classification
The firstly implemented classification method for the project is the Naive Bayes Method. In order to have a better classification result both the train data and test data were normalized beforehand. These normalized data were used in the Naive Bayes Method. For this method Gaussian probability distribution was utilized.

Gaussian Probability Distribution:

![Screenshot 2023-12-23 153238](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/8e16dc71-95c2-4bb8-8af8-fd3de5896c79)

Eqn. 2: Gaussian Probability Distribution

Naïve Bayes method performs under the assumption that features are independent from each other. A probability is assigned to every feature. As it was mentioned before, we used Gaussian Probability Distribution to assign these values. Then, we integrated this formula with Bayes’ Theorem.

Bayes’ Theorem:

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/f2de8de5-b168-42f6-b282-5e0d6b12c15a)

Eqn. 3: Bayes’ Theorem

We can also write this as the following:

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/d186f751-b54c-41eb-99f2-2484646fb009)

Eqn. 4: Bayes’ Theorem

Since we assumed independence of variables, by multiplying probabilities of every column for every row we reach the likelihood. However, when we multiplied the probabilities, we acquired some errors. Since the Gaussian formula uses exponential, overflow occurred in some rows. In order to overcome this issue, we have taken logarithm of these values and instead of multiplication we used summation in every row. By summing up these values, we acquired the likelihood of each and every row. We used such a process because we know that by taking logarithm, we are changing the final values proportionally. Also, we didn’t use ‘evidence’ probability because we would have the same denominator for every calculation and it would have cancelled itself.

                        𝑝𝑜𝑠𝑡𝑒𝑟𝑖𝑜𝑟 𝖺 𝑙𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑 × 𝑝𝑟𝑖𝑜𝑟
                        Eqn. 5: Ratio of Probabilities

We defined a function called ‘naïve_prob_generate’ to calculate the probabilities of the dataset. It takes the dataset and related mean and standard deviation values as input and returns the probabilities of every row. Then, by using these values, we classified the test dataset. After training the algorithm we created the confusion matrix which can be seen under the result section.

Logistic Regression

As another method to classify the examined dataset, logistic regression is preferred. To obtain classification results, the normalized train dataset is utilized. The features in the train data are multiplied with weights and a bias is added. In each step weights are updated by stochastic gradient descent algorithm. At the end of a specified number of iterations, optimum weights are reached. The formula for the weighted sum is given as below.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/654a08b6-3561-4867-bb02-ed9890cda0f8)

Eqn. 6: Weighted Sum

Obtaining the z, the weighted sum, sigmoid function is used as we deal with logistic regression. With the help of sigmoid function, the data becomes cumulated as it is demonstrated in the following figure. The predictions that are spreaded in the range [0, 1] become 0 or 1 approximately. So, the shown cumulation of data points after sigmoid represent more certain classification results compared to linear regression.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/838a2b67-1817-428a-b3d4-9a5e59b2eb4a)

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/6153b936-87f9-401a-b7fa-a30549bc071c)

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/b62d7388-8b59-40d4-a19e-466061304413)

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/86c2870f-6d24-4322-8be5-73fdcfb62a6b)

Figure 4: Sigmoid curve [4]

In the calculation of updated weights, the main objective is to minimize the loss function. So, the derivative of the loss is inspected to reach the optimum result. First we tried to reach a result without taking the logarithm of the loss function and calculations resulted with overflow. Then logarithm of the loss is utilized to handle the issue. However, loss function is not enough to derive the best result. Various learning rates and iteration numbers are examined, then the values that lead the best result are chosen in the implementation of the method. Applying these steps, weights are updated with respect to the stochastic gradient descent algorithm as stated in the following equations.

𝐿 (𝑤, 𝑏) =	− [(1 − 𝑦) 𝑙𝑜𝑔(1 − σ(𝑤 · 𝑥 + 𝑏)) + 𝑦 𝑙𝑜𝑔σ(𝑤 · 𝑥 + 𝑏)]
Eqn. 9: Loss Function

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/9a9ea443-4b9e-44d6-a773-862662f5f3be)

So, we implemented the logistic regression method by following the mentioned steps after initializing the weights. Training the data, accuracy is calculated on the test set by comparing predictions and original labels.

As overflow should be avoided in calculations, normalization is utilized. It is expected to obtain better results compared to Naive Bayes, as it is more complicated and takes more time to compute. Also, in this part we have used a validation set to decide on which learning rate is the best and what epoch number results with the highest accuracy. After training we created the confusion matrix for the test set. It can be seen in the result section.

Decision Tree
 
Decision tree used for the third classification method for our project which is a non_parametric supervised learning method. The purpose for using this model is to generate a model which is used for the prediction of the round winner label by creating a tree structure and learn the decision rules that are related to our dataset and our features. The values of the attributes are used to split the data into nodes. There is a two child node generated for each iteration over the features and the possible split points of those features from the iterated feature. The effectiveness of the split point and children is determined by using the cross entropy formula which is given in the following figure.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/13d6f291-a701-4c37-a461-272fc3e8ab35)

Figure 5: Cross Entropy Formula[6]

In the formula, In the mth branch of the tree the probability of the class k is used. The result of the entropy formula would take a small value when the given child is pure. Later, the average cross entropy is calculated by multiplying cross entropy values of each child with the probability of one sample being in the child. So, when the cross entropy average is the smallest, it will be the best split result. Finding the best splits continues until the tree is over and is pure. The tree structure would be like in the following figure.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/67b7fc40-cb7b-43c2-af01-2c7dea3ca7c4)

Figure 6: The decision tree structure[7]
In the decision tree model, creating the tree is the training structure. So, we find the tree by using the train set of our data. Once the tree structure is formed, the machine learning algorithm is ready for the testing procedure. The most significant part of the decision tree algorithm, when the stucture is pure too much, the algorithm will overfit which means when testing the algorithm, there will be too little accuracy which we do not want.

Simulation Results
After training with Naive Bayes Method, we created the confusion matrix. From this matrix we acquired the precision, accuracy, and recall values. The accuracy of Naive Bayes Method varied between 50-70%. This variation is caused by the dataset shuffling. Since we do not have a specific shuffled training set, it changes for every code execution. Depending on the train and test data separation, we get an accuracy value in the 50-70% interval. The precision value is very close to the accuracy value, it varies accordingly. After many tries, we have seen that the accuracy value does not get lower than 50%. The independence assumption leads to reduction in accuracy also. We have many features. There is a high probability that at least few of these are dependent on each other. However, our algorithm is very economical in terms of time and resource usage. Training the algorithm only
 
takes 0.11 seconds and the testing takes 0.15 seconds. Both training and testing does not even add up to 1 second. Therefore, we are satisfied with the result of Naïve Bayes method.

In the Logistic Regression Method, we have used a validation set to determine the best parameters which give the best accuracy. For different learning rates and epoch numbers we run tests with validation set. Then, we have seen that the best result was acquired when the learning rate is 0.01 and epoch number is 2. When we increased the epoch number we have seen that the accuracy converges around 65%. The best accuracy was 75%. However, same as the Naive Bayes Method, our result differs depending on the dataset division. Following confusion matrices are an instance of a random shuffling.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/8aaea239-facd-4fdc-bdac-a77027c46152)

From the previous graphs it can be seen that the learning rate has a very minimal effect on the accuracy. The epoch number is more influential on the result.
In terms of run time, the Logistic Regression took 5.46 seconds to train and 0.19 seconds to test. In contrast training took longer than Naive Bayes because we looked for the best parameters by running for 100 iterations and 5 different learning rates. At first we used two nested for loops to do such training and it took even longer. However, by using matrix operations we got rid of the second for loop and trained the algorithm for only the iteration number.

![image](https://github.com/MehmetOguzhanTor/CsGoRoundWinnerClassifier/assets/116079107/91b18a77-808c-485e-8170-d8dd20536c94)

In the Decision Tree method, we have used the train set in order to train the algorithm. In this case, there is a tree structure that is formed during the process with the help of calculations that are mentioned in the above sections. In the decision tree algorithm, we used the maximum depth of the main function which is the depth of the tree structure to be 7 after some trials. When the test is run with the test data, we get the accuracy between 48 and 52. The accuracy seems low with comparison with the first two methods. The reason for this would be the algorithm being overfit or underfit with the tree structure. In the Decision tree algorithm the run time is 29.6 (s) which is higher than the first methods. It is because creating a tree takes more time. Yet, the time is not as high as we expected.

Discussion on the Performance
One of the challenges about the project was arranging the dataset. It took more time and effort than we anticipated. In order to work with the dataset, we need to always shuffle and consider the all 0-columns at all times which is learned from the hard way because we faced some errors caused from this situation and tried to find out why.
When we classified the dataset by using Naive Bayes Method, the accuracy resulted in a maximum value of 70.62% which can be considered as a satisfying result. In addition to accuracy, precision and recall was calculated. The precision was around 72% and the recall was around 55%. In terms of time performance, Naive Bayes was very successful. The training and the testing took less than 1 second (~0.3 seconds).
In the Logistic Regression Method the accuracy was higher than Naive Bayes. Generally Logistic Regression had 5% higher accuracy than Naive Bayes. The precision was almost 10% higher than naive Bayes’. The recall values are very similar. In terms of time performance, it took 5 seconds to
 
train the algorithm and 0.2 seconds to test. It took more than Naive Bayes because we used 100 iterations for the validation set. From the validation set we determined the best learning rate and epoch number. With these values it took less than 1 second to test our algorithm.
Logistic Regression had better result in terms of accuracy because of the independence assumption of the Naive Bayes Method.

In the last model of the project, the most challenging thing was not overfitting the decision tree structure. It is because, when there is overfitting in the algorithm, the accuracy of our testing drops dramatically. After the trial of some parameter of maximum depth of decision tree structure of number of samples, we have reached the optimum accuracy of 48-52. However, still it may be overfitting considering that the accuracy is lower than the other two methods.

References
[1]	Lillelund, Christian. “CS:GO Round Winner Classification.” Kaggle, 19 Aug. 2020, www.kaggle.com/christianlillelund/csgo-round-winner-classification.
[2]	TOR, Mehmet O, TAŞDEMİR Berkin., “EEE 485 Statistical Learning and Data Science Project Proposal CS:GO Round Winner Classification”. 2021.
[3]	Chatterjee, Soumo. “Use Naive Bayes Algorithm for Categorical and Numerical Data Classification.	”Medium,	26	Nov.	2020, medium.com/analytics-vidhya/use-naive-bayes-algorithm-for-categorical-and-numerical-data-cl assification-935d90ab273f.
[4]	L. Su, “Logistic Regression, Accuracy, Cross-Validation,” Medium, 14-May-2019. [Online]. Available:
https://medium.com/@lily_su/logistic-regression-accuracy-cross-validation-58d9eb58d6e6. [Accessed: 06-May-2021].
[5]	D. Jurafsky and J. H. Martin, “Speech and Language Processing,” 30-Dec-2020. [Online]. Available: https://web.stanford.edu/~jurafsky/slp3/5.pdf.
[6]	“Loss Functions¶,” Loss Functions - ML Glossary documentation. [Online]. Available: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html. [Accessed: 07-May-2021].
[7]	N. Tyagi, “Decision Tree in Machine Learning,” Analytics Steps. [Online]. Available: https://www.analyticssteps.com/blogs/decision-tree-machine-learning. [Accessed: 07-May-2021].
