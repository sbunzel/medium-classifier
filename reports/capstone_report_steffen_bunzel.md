# Machine Learning Engineer Nanodegree
## Capstone Project
Steffen Bunzel
November 25th, 2018

## I. Definition

### Project Overview
This project evaluates the use of machine learning to tackle the omnipresent problem of [information overload](https://en.wikipedia.org/wiki/Information_overload).
In particular, it targets the ever growing amount of textual information on the web in the form of blog posts.
Blog posts can be a great source of information on a wide variety of specialized topics. They are commonly more accessible than books
and many provide practical insights from experts free of charge. However, there is a large spread in quality between well-written, informative and low quality "clickbait" posts ([Gardiner, 2015](https://www.wired.com/2015/12/psychology-of-clickbait/)). Especially in "hot" fields, many authors try to benefit from the hype by putting out posts with catchy titles to impress newcomers and gather clicks without sharing any insightful content.

One of these is the field of machine learning itself. This area has received increasing attention in recent years (compare Figure 1). As a consequence, the topic is covered extensively in both popular media and by technical writers.

<center><img src="./images/interest-in-ml.png" alt="Interest in ML" width="400"/></center>
<center>*Figure 1: Interest in machine learning as measured by Google Trends*
<br>
<br>
</center>

While this growing interest has certainly fueled investment in promising technologies powered by machine learning, it also makes it harder to separate the signal from the noise. The remainder of this report details an approach that leverages machine learning to tackle this problem. To do so, a system is developed that automatically suggests whether or not the user will find a particular machine learning blog post interesting based on the post's text.

### Problem Statement
In order to delineate well-defined objectives and benchmarks, the problem space is narrowed down by imposing three central restrictions:

1. The focus is on machine learning blog posts published at [medium.com](https://medium.com/).
2. These posts are grouped in two categories: "Interesting" and "Not interesting". These will be what machine learning is used to differentiate between. The machine learning task will therefore be *binary classification*.
3. The definition of "interesting" and the prioritization of the two types of error a binary classification model can make (1. Cluttering the results by wrongly predicting an uninteresting post to be interesting vs. 2. missing an interesting post by wrongly predicting it as uninteresting) is done based on the authors preferences (see section on *Metrics* below).

Of course, restriction 3) does not imply that the approach is only relevant to the authors preferences. The general framework can be tailored to anyone by substituting the input labels and adjusting the metric prioritization based on their liking.

To illustrate this, compare an exemplary problem statement tailored to the author:

> I love to read blog posts on machine learning (ML) and artificial intelligence (AI) on medium.com, a popular online publishing platform. However, not all of the articles available there fit my taste. Unfortunately, I sometimes find this only after having read the article. In particular, I do not enjoy articles which are superficial and aim to benefit from the hype around ML and AI, thereby confusing beginners. On the other hand, I enjoy articles offering a deep coverage of technical concepts and are written in a way that keeps the reader engaged nevertheless. As I am always short on time, I am particularly interested in a solution that reliably identifies blog posts that I will really enjoy as interesting to me and would be willing to sacrifice a few articles that I might have liked, but which the model misses.

### Metrics
The metrics that will be used to assess the classifier's performance need to balance two aspects: The user has limited capacity to read blog posts and wants to find the ones they are interested in with high precision. Thus, the first metric employed is precision:

`precision = true positives / (true positives + false positives)`

Based on the author's problem statement above, a **target precision of 95%** will serve as a guideline for this project, which means that only one in twenty posts recommended by the classifier are actually uninteresting.

On the other hand, the number of article recommendations should not be restricted too much. Therefore, the model's ability to identify a substantial share of the actually interesting posts as such must be taken into consideration as well. This objective is covered by the `recall` metric. As the author prioritizes precision over recall, the objective for this metric is set to a **recall of 75%**, which means that only one in four interesting posts are classified as not interesting:

`recall = true positives / (true positives + false negatives)`

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
To tackle this problem, a dataset of ML blog posts published on medium.com is needed.
Luckily, a Kaggle user has already provided a collection of such data on the [platform](https://www.kaggle.com/hsankesara/medium-articles/home). This dataset was scraped from medium.com and made available under the `CC0: Public Domain license, i.e. as a work in the public domain without copyright` license. The dataset contains the post's author, its estimated reading time, a link to the post, its title, its text body as well as the number of claps (a form of like) it has received to date.

While the raw data contained 337 rows, it included several duplicates, resulting in 230 unique articles. Of these, six were not written in English, thus reducing the number to 224. To keep the requirements with respect to provisions from the user manageable, the top 130 posts based on their number of claps were labeled as either "interesting" or "uninteresting" by the author after manually assessing each post on medium.com for 2-5 minutes.

Thus, the data contains 130 observations of 5 potential features (ignoring the link to the article) and one target variable ("interesting" or "uninteresting"). Two of these features are numerical: Claps and reading time.

<center>
<table border="2" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th><center></th>
      <th><center>claps</th>
      <th><center>reading_time</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <th><center>count</th>
      <td>130.00</td>
      <td>130.00</td>
    </tr>
    <tr align="center">
      <th><center>mean</th>
      <td>7064.05</td>
      <td>10.67</td>
    </tr>
    <tr align="center">
      <th><center>std</th>
      <td>9836.08</td>
      <td>5.61</td>
    </tr>
    <tr align="center">
      <th><center>min</th>
      <td>10.00</td>
      <td>3.00</td>
    </tr>
    <tr align="center">
      <th><center>25%</th>
      <td>1500.00</td>
      <td>7.00</td>
    </tr>
    <tr align="center">
      <th><center>50%</th>
      <td>3300.00</td>
      <td>9.00</td>
    </tr>
    <tr align="center">
      <th><center>75%</th>
      <td>8450.00</td>
      <td>13.00</td>
    </tr>
    <tr align="center">
      <th><center>max</th>
      <td>53000.00</td>
      <td>31.00</td>
    </tr>
  </tbody>
</table>
*Table 1: Summary statistics on the numerical features*
<br>
<br>
</center>

The other columns all contain non-numeric values and are almost always unique.
<center>
<table border="2" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th><center></th>
      <th><center>author</th>
      <th><center>title</th>
      <th><center>text</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <th><center>count</th>
      <td>130</td>
      <td>130</td>
      <td>130</td>
    </tr>
    <tr align="center">
      <th><center>unique</th>
      <td>102</td>
      <td>130</td>
      <td>130</td>
    </tr>
    <tr align="center">
      <th><center>top</th>
      <td>Adam Geitgey</td>
      <td>Distributed Neural Networks with GPUs in the A...</td>
      <td>What Machine Learning Teaches Us About Ourselv...</td>
    </tr>
    <tr align="center">
      <th><center>freq</th>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
*Table 2: Summary statistics on the non-numerical features*
<br>
<br>
</center>

These basic statistics show that the data underlying this project has a simple structure: It consists of the contents of a blog post (represented by its title and text body) as well as three pieces of metadata (the post's author, the number of claps and the estimated reading time).

As mentioned above, the target variable of this dataset is whether the author found the particular article interesting or not after assessing it manually for 2-5 minutes. This variable is binary: It is 0 for posts the author found uninteresting and 1 for posts he found interesting. The distribution of the classes is slightly imbalanced: Approximately 35% of the posts are labeled "interesting" while the remaining 65% are labeled "uninteresting".

### Exploratory Visualization

<center><img src="./images/base_classifier.png" alt="Classes by claps and reading time" width="400"/></center>
<center>*Figure 2: Classes by claps and reading time*
<br>
<br>
</center>


In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

*Sources:*

- *Gardiner (2015): Gardiner, Bryan (2015) You'll be Outraged at How Easy it Was to
Get You to Click On This Headline. Available at: https://www.wired.com/2015/12/psychology-of-clickbait/ (Last accessed: 2018-11-25).
