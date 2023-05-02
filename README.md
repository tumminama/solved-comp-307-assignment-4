Download Link: https://assignmentchef.com/product/solved-comp-307-assignment-4
<br>






<h1>Part 1 : Classical Planing</h1>

<strong>Question 1: </strong>

Initial state:

At(Monkey,A) ∧ At(Box,B) ∧ At(Bananas,C) ∧ Height(Monkey,Low) ∧ Height(Box,Lox) ∧ Height(Banana,High)

Goal state:

Hold(Monkey,Banana)

<strong>Question 2: </strong>Lowercase = variable || Capital letters = value

Go(X,Y)

Precondition: At(Monkey,X) <sub>∧</sub> ¬Height(Monkey,High) <sub>∧</sub> (X≠ Y)

Effect: At(Monkey,Y) <sub>∧</sub> ¬At(Monkey,X)

Push(Box,X,Y)

Precondition: At(Box,X) ∧ At(Monkey,X) ∧ Height(Monkey,Low) ∧ (X≠ Y)     Effect: At(Box,Y) <sub>∧</sub> At(Monkey,Y)

ClimbUp

Precondition: At(Monkey,X) <sub>∧</sub> At(Box,X) <sub>∧</sub> Height(Monkey,Low)

Effect: At(Monkey,X) ∧ At(Box,X) ∧ Height(Monkey,High) ∧                                            ¬Height(Monkey,Low)

ClimbDown

Precondition: At(Monkey,X) <sub>∧</sub> At(Box,X) <sub>∧</sub> Height(Monkey,High)

Effect: At(Monkey,X) ∧ At(Box,X)∧ Height(Monkey,Low) ∧                                              ¬Height(Monkey,High)

Grasp(X,h)

Precondition: At(Monkey,X) ∧ At(Banana,X) ∧ Height(Monkey,h) ∧

Height(Bananas)   Effect: Hold(Monkey,Bananas)

Ungrasp(Bananas)

Precondition: Hold(Monkey,Bananas)

Effect: ¬Hold(Monkey,Bananas)

<strong>Question 3: </strong>

<strong> </strong>

<strong>Question 4: </strong>

Initial state:  At(Monkey,A) ∧ At(Box,B) ∧ At(Bananas,C) ∧ Height(Monkey,Low) ∧

Height(Box,Lox) <sub>∧</sub> Height(Banana,High) <sub>∧</sub> ¬Hold(Monkey,Banana)

Action1: Go(A,B)

State1: At(Monkey,B) ∧ At(Box,B) ∧ At(Bananas,C) ∧ Height(Monkey,Low) ∧ Height(Box,Lox) <sub>∧</sub> Height(Banana,High)

Action2: Push(Box,B,C)

State2: At(Monkey,C) ∧ At(Box,C) ∧ At(Bananas,C) ∧ Height(Monkey,Low) ∧

Height(Box,Lox) <sub>∧</sub> Height(Banana,High) <sub>∧</sub> ¬Hold(Monkey,Banana)

Action3: ClimbUp(C)

State3: At(Monkey,C) ∧ At(Box,C) ∧ At(Bananas,C) ∧ Height(Monkey,High) ∧

Height(Box,Lox) <sub>∧</sub> Height(Banana,High) <sub>∧</sub> ¬Hold(Monkey,Banana)

Action4: Grasp(C,High)

Goal state: At(Monkey,C) ∧ At(Box,C) ∧ At(Bananas,C) ∧ Height(Monkey,High) ∧ Height(Box,Lox) <sub>∧</sub> Height(Banana,High) <sub>∧</sub> Hold(Monkey,Banana)

<h1>Part 2 : Job Shop Scheduling</h1>

<strong>Question 1: </strong>

Process(O<sub>11</sub>, M<sub>1</sub>, t<sub>1</sub>) → Process(O<sub>21</sub>, M<sub>2</sub>, t<sub>2</sub>) → Process(O<sub>31</sub>,M<sub>1</sub>,t<sub>3</sub>) → Process(O<sub>12</sub>,M<sub>2</sub>,t<sub>4</sub>) → Process(O<sub>22</sub>,M<sub>1</sub>,t<sub>5</sub>) → Process(O<sub>32</sub>,M<sub>2</sub>,t<sub>6</sub>).

1.

Job ready time:

J

J

J3:    O31   = +∞     O32 = +∞<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 0

M<sub>2</sub> = 0

Operation(O<sub>11</sub>, M<sub>1</sub>, ProcessTime = 0) = -50

t<sub>1</sub> = 0

2.

Job ready time:

J1:    <span style="text-decoration: line-through">O</span>11   <span style="text-decoration: line-through">= 0</span>     O12 = 50

J

J3:    O31   = 20     O32 = +∞<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 50

M<sub>2</sub> = 0

Operation(O<sub>21</sub>, M<sub>2</sub>, t<sub>2</sub>) = Operation(O<sub>21</sub>, M<sub>2</sub>, ProcessTime = 10) = -30     t<sub>2</sub> = 10

3.

Job ready time:

J1:    <span style="text-decoration: line-through">O</span>11   <span style="text-decoration: line-through">= 0</span>     O12 = 50

J2:    <span style="text-decoration: line-through">O</span>21   <span style="text-decoration: line-through">= 10</span>     O12 = 40

J3:    O31   = 20     O32 = +∞<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 50

M<sub>2</sub> = 40

Operation(O<sub>31</sub>, M<sub>2</sub>, ProcessTime) = -40     t<sub>3</sub> = 50

4.

Job ready time:

J1:    <span style="text-decoration: line-through">O</span>11   <span style="text-decoration: line-through">= 0</span>     O12 = 50

J2:    <span style="text-decoration: line-through">O</span>21   <span style="text-decoration: line-through">= 10</span>     O12 = 40

J3:    <span style="text-decoration: line-through">O</span>31   <span style="text-decoration: line-through">= 20</span>     O32 = 90<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 90

M<sub>2</sub> = 40

Operation(O<sub>12</sub>, M<sub>2</sub>, ProcessTime) = 25     t<sub>4</sub> = 50

5.

Job ready time:

J1:    <span style="text-decoration: line-through">O</span>11   <span style="text-decoration: line-through">= 0</span>     <span style="text-decoration: line-through">O</span>12 <span style="text-decoration: line-through">= 50</span>

J2:    <span style="text-decoration: line-through">O</span>21   <span style="text-decoration: line-through">= 10</span>     O12 = 40

J3:    <span style="text-decoration: line-through">O</span>31   <span style="text-decoration: line-through">= 20</span>     O32 = 90<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 90

M<sub>2</sub> = 75

Operation(O<sub>22</sub>, M<sub>1</sub>, ProcessTime=90) = -35     t<sub>5</sub> = 90

6.

Job ready time:

J3: <sub> </sub>O32 = 90<strong> </strong>

Machine Idle time:

M<sub>1</sub> = 125

M<sub>2</sub> = 75

Operation(O<sub>32</sub>, M<sub>2</sub>, ProcessTime=90) = -20     t<sub>6</sub> = 90

t<sub>1</sub> = 0, t<sub>2</sub> = 10, t<sub>3</sub> = 50, t<sub>4</sub> = 50, t<sub>5</sub> = 90, t<sub>6</sub> = 90 So      t1 &lt; t2 &lt;t3 = t4 &lt; t5 = t6

<strong>Question 2: </strong>

Finishing time of J<sub>1</sub> = t<sub>4</sub> + processTime(O<sub>12</sub>) = 75

Finishing time of J<sub>2</sub> = t<sub>5</sub> + processTime(O<sub>22</sub>) = 125

Finishing time of J<sub>3</sub> = t<sub>6</sub> + processTime(O<sub>32</sub>) = 110

The makespan of this solution is the time of the job finished latest which is 125.

<strong>Question 3: </strong>

<u>Step1:</u>

Partial solution : Process(O<sub>11</sub>, M<sub>1</sub>, ProcessTime=0)      earliest Idle time(M<sub>1</sub>)=50, earliest Idle time(M<sub>2</sub>)=0       earliest ready time (O<sub>12</sub>)=50, earliest ready time (O<sub>21</sub>)=10, earliest ready time (O     earliest ready time (O<sub>31</sub>)=20, earliest ready time (O

<u>Step2:</u>

Partial solution : Process(O<sub>11</sub>, M<sub>1</sub>, ProcessTime=0) &#x27a1; Process(O<sub>21</sub>, M<sub>2</sub>, ProcessTime=10) earliest Idle time(M<sub>1</sub>)=50, earliest Idle time(M<sub>2</sub>)=40

earliest ready time (O<sub>12</sub>)=50,         earliest ready time (O<sub>22</sub>)=40,

earliest ready time (O<sub>31</sub>)=20, earliest ready time (O

<u>Step3:</u>

Partial solution : Process(O<sub>11</sub>, M<sub>1</sub>, ProcessTime=0) &#x27a1; Process(O<sub>21</sub>, M<sub>2</sub>, ProcessTime=10)

<ul>

 <li>Process(O<sub>12</sub>, M<sub>2</sub>, ProcessTime=50)</li>

</ul>

earliest Idle time(M<sub>1</sub>)=50, earliest Idle time(M<sub>2</sub>)=75       earliest ready time (O<sub>22</sub>)=40,

earliest ready time (O<sub>31</sub>)=20, earliest ready time (O

<u>Step4:</u>

Partial solution : Process(O<sub>11</sub>, M<sub>1</sub>, ProcessTime=0) &#x27a1; Process(O<sub>21</sub>, M<sub>2</sub>, ProcessTime=10)

<ul>

 <li>Process(O<sub>12</sub>, M<sub>2</sub>, ProcessTime=50) &#x27a1; Process(O<sub>22</sub>, M<sub>1</sub>, ProcessTime=50)</li>

</ul>

…….

<u>Final solution:</u>

Process(O<sub>11</sub>, M<sub>1</sub>, ProcessTime=0) &#x27a1; Process(O<sub>21</sub>, M<sub>2</sub>, ProcessTime=10) &#x27a1; Process(O<sub>12</sub>,

M<sub>2</sub>, ProcessTime=50) &#x27a1; Process(O<sub>22</sub>, M<sub>1</sub>, ProcessTime=50) &#x27a1; Process(O<sub>31</sub>, M<sub>1</sub>, ProcessTime=85) &#x27a1; Process(O<sub>32</sub>, M<sub>2</sub>, ProcessTime=125)

<strong>Question 4: </strong>

The completion time of J<sub>1 </sub>= 75

The completion time of J<sub>2 </sub>= 85

The completion time of J<sub>3 </sub>= 145

The makespan of SPT is MakeSpan(J<sub>1</sub>, J<sub>2</sub>, J<sub>3</sub>) = 145.

Comparing the makespans of FCFS with SPT, 125&lt;145, in this case, FCDS is better than

SPT in makespans.

<strong>Question 5: </strong>

No, it doesn’t. One solution is better than the other does not mean that the rule that generates the better solution is better than the other rule. It just means one rule is more appropriate than the other rule in this case.

<h1>Part 3 : Search Techniques and Machine Learning Basics: Questions</h1>

<strong>Question1: </strong>

1.

<ul>

 <li>Depth-first search. Since the cost from the intermediate state to the target state is unknown, depth-first search can help search from the intermediate state to the target state.</li>

 <li>Breadth first search. Because the cost of going from the initial state to the intermediate state is unknown, breadth first search is suitable for searching the intermediate state.</li>

 <li>Heuristic search. Heuristic functions estimate the cost of going from the current state to the target state.</li>

</ul>

2.

For workshop issues, local optimal issues is a problem, sometimes we need to get worse local result first and then get best final result in the end. Heuristic search —genetic beam search can help us jump out of local optima and get a better result. 3.

<ul>

 <li>Breadth first search : 1,2,3,4,5,6,7,8,9,10,11,12,13</li>

 <li>Iterative deepening search :</li>

</ul>

Limit = 0

1

Limit = 1

1,2,3

Limit = 2

1,2,4,5,3,6,7

Limit = 3

1,2,4,8,9,5,10,11,3,6,12,13,7

<strong>Question 2: </strong>

1.

<u>Possible reason 1:</u>

The choice of k has a great influence on the kNN learning model. If the value of k is too small, the prediction result will be extremely sensitive to noise sample points. In particular, when k is equal to 1, kNN degenerates into the nearest neighbor algorithm, and there is no explicit learning process. If the value of k is too large, there will be larger neighborhood training samples for prediction, which can reduce the reduction of noise sample points; but the training sample points that are farther away will contribute to the prediction result, so that the prediction result will be wrong.

<u>Solution:</u>

Let Michael try different odd k values

<u>Possible reason 2:</u>

The training set is too small, and those domains with smaller sample sizes are more prone to misclassification using this algorithm.

<u>Solution:</u>

Let Michael select more instances to join the training set while ensuring a balanced number of categories of instances.

2.

<u>Possible reason 2:</u>

I think it is probably overfitting.

<u>Solution:</u>

We can do early stop or pruning.

<strong>Question 3: </strong>

Impurity = P(popular) <sub>∗</sub> P(unpopular)

Feature 1: Mushroom

Yes Impurity =1/5 × 4/5 = 4/25

No Impurity =4/5 × 1/5 = 4/25

Weighted Average Impurity = 4/25 = 16%

Feature 2: Vegetarian

Yes Impurity =1/4 × 3/4 = 3/16

No Impurity =4/6 × 2/6 = 2/9

Weighted Average Impurity = 0.4 × 3/16 + 0.6 ×2/9 = 20.83%

Feature 3: Size

Small Impurity =1/3 × 2/3 = 2/9

Medium Impurity =2/3 × 1/3 = 2/9

Large Impurity =2/3 × 1/3 = 2/9

Weighted Average Impurity = 2/9 = 22.22%

Mushroom is the best feature, it should be chosen for the root of the tree.

<h1>Part 4 : Other Topics: Questions</h1>

<ol>

 <li>Deep learning is an <a href="https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp">artificial intelligence</a> function that imitates the workings of the human brain in processing data and creating patterns for use in decision making. Deep learning is a subset of <a href="https://www.investopedia.com/terms/m/machine-learning.asp">machine learning</a> in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep neural network.</li>

</ol>




Algorithms and Examples:

<ol>

 <li>CNN (Convolutional neural networks) :</li>

</ol>

CNN have been successful in identifying faces, objects, and                                      traffic signs apart from powering vision in robots and self-driving

cars.

<ol start="2">

 <li>BP (Back propagation) :</li>

</ol>

Let’s say that for some reason we want to identify images with a tree.                                 We feed the network with any kind of images and it produces an                                          output. Since we know if the image has actually a tree or not, we can                                  compare the output with our truth and adjust the network. As we                                                       pass more and more images, the network will make fewer and fewer

mistakes.

<ol start="3">

 <li>MLP (Multilayer Perceptron Neural Network) :</li>

</ol>

Image verification and reconstruction; Speech recognition; Machine                                    translation; Data classification.

<ol start="2">

 <li>In <a href="https://en.wikipedia.org/wiki/Machine_learning">machine learning,</a> support-vector machines (SVMs, also support-vector networks) are <a href="https://en.wikipedia.org/wiki/Supervised_learning">supervised learning</a> models with associated learning <a href="https://en.wikipedia.org/wiki/Algorithm">algorithms</a> that analyze data used for <a href="https://en.wikipedia.org/wiki/Statistical_classification">classification</a> and <a href="https://en.wikipedia.org/wiki/Regression_analysis">regression analysis</a>.</li>

</ol>

When data is linear separable, you can use a support vector machine (SVM) when your data has exactly two classes. An SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class.

The best hyperplane (the yellow part of the graph below) for an SVM means the one with the largest margin between the two classes.

<em>Reprinted from “Support vector machine,” by Wikipedia </em>

<em>                 </em>For those non-linear separable data I would like to mention is from assignment 2,

For three-dimensional data like this, we cannot find a plane to distinguish the

points of the two colors. <em>(Graph from my own assignment2) </em>

<ol start="3">

 <li>Text mining :</li>

</ol>

Example application :

Security applications: Many text mining software packages are                                             marketed for security applications, especially monitoring and analysis                                 of online plain text sources such as Internet news, blogs, etc. for                                                          national security purposes. It is also involved in the study of text

encryption/decryption.




Algorithm :

Bag of Words (BoW): BoW is all about creating a matrix of words                                         where the words (terms) are represented as the rows and the                                               columns represent the document names. We can then populate the

matrix with the frequency of each term within the document, ignoring                                              the grammar and order of the terms.

Natural language processing :

Example application :

Companies like Yahoo and Google filter and classify your emails with                                 NLP by analyzing text in emails that flow through their servers                                                          and stopping spam before they even enter your inbox.

Algorithm :

Naive Bayes :

In most cases, NBA in the Natural Language Processing sphere is                                       used for text classification (clustering). The most known task is a                                          spam detection filter. Most solutions in this sphere use the maximum                                   likelihood method to estimate the parameters of Naive Bayesian

models:




The first multiplier defines the probability of the text class, and the                                        second one determines the conditional probability of a word

depending on the class.




<ol start="4">

 <li></li>

</ol>

<u>An expert system</u> is software that attempts to provide an answer to a problem, or clarify uncertainties where normally one or more human experts would need to be consulted. Expert systems are most common in a specific problem domain, and is a traditional application and/or subfield of artificial intelligence.

<u>A Decision Support System (DSS)</u> is a class of information systems (including but not limited to computerized systems) that support business and organizational decisionmaking activities. A properly designed DSS is an interactive software-based system intended to help decision makers compile useful information from a combination of raw data, documents, personal knowledge, or business models to identify and solve problems and make decisions.




In the medical field, a KBS can help doctors more accurately diagnose diseases. These systems are called <u>clinical decision-support systems</u> in the health industry.

5.

<strong>Volume</strong> defines the huge amount of data that is produced each day by companies, for example. The generation of data is so large and complex that it can no longer be saved or analyzed using conventional data processing methods.

<strong>Variety</strong> refers to the diversity of data types and data sources. 80 percent of the data in the world today is unstructured and at first glance does not show any indication of relationships. Thanks to Big Data such algorithms, data is able to be sorted in a structured manner and examined for relationships. Data does not always comprise only conventional datasets, but also images, videos and speech recordings.

<strong>Velocity</strong> refers to the speed with which the data is generated, analyzed and reprocessed. Today this is mostly possible within a fraction of a second, known as real time.

<strong>Validity</strong> is the guarantee of the data quality or, alternatively, Veracity is the authenticity and credibility of the data. Big Data involves working with all degrees of quality, since the Volume factor usually results in a shortage of quality.

<strong>Value</strong> denotes the added value for companies. Many companies have recently established their own data platforms, filled their data pools and invested a lot of money in infrastructure. It is now a question of generating <a href="https://blog.unbelievable-machine.com/en/blog/trends-in-2016-the-next-big-things-in-the-data-world-part-1/">business value</a> from their investments.