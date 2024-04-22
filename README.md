# Fairness in Machine Learning for 2nd Order Solutions

#### Contributors: [Yiyu Lin](https://github.com/YYLinn), [Yinting Zhong](https://github.com/YintingZhong), [Genesis Qu](https://github.com/qu-genesis), [Pragya Raghuvanshi](https://github.com/pr-124)

## Table of Contents
1. [Introduction](#Introduction)
2. [Project Objective & Goals](#Goals)
3. [Datasets](#Datasets)  
4. [Overall Methodology](#Methodology)  
5. [Results](#Results)
6. [Conclusion](#Conclusion)
7. [Usage](#Usage)



## 1.Introduction <a name="Introduction"></a>
2nd Order Solutions, our client in this capstone project, is a financial consulting firm that works on providing analytical solutions to their financial partners – mainly banks domestically and internationally. The company uses most of its time to build statistical models to help clients craft valuation and credit lending policies, fraud detection, and due diligence.
<br>As an institution that provides financial services to the public, 2OS, and its clients operate under a strict network of regulatory frameworks and oversight bodies. A key aspect of such regulation is the requirement – under the [Equal Credit Opportunity Act](https://consumer.ftc.gov/articles/credit-discrimination) – that the models that decide what consumers receive financial products may not discriminate on protected characteristics of the clients such as gender, race, disability status, and ethnicity. Such requirements are fundamental to the service that 2OS provides because current regulations render any model that introduces biases unusable.
How machine learning algorithms perpetuate bias is keenly researched in academia and the tech media world. A frequent way bias shows up is through biased training data. For example, if most women were denied opportunities in a company while few men were, then an algorithm trained on this data to screen resumes would doubtlessly recommend men disproportionately. In sensitive fields such as healthcare and finance, such bias needs to be carefully guarded against. Our goal is to provide 2OS with tools to assess fairness and mitigate the biases before model handover, enhancing their business processes and value proposition.


## 2.Project Objective & Goals <a name="Goals"></a>
The purpose of our capstone team and this project is to research the evaluation of fairness in financial machine-learning products and evaluate current packages that quantify algorithmic bias in models. We define fairness as an equal opportunity to obtain a positive outcome for both the underprivileged and the privileged groups. The goal is to make recommendations to 2OS  on which statistical package(s) best fulfills its need to remain compliant with financial regulations.
The purpose of our capstone team and this project is to research the evaluation of fairness in financial machine-learning products and evaluate current packages that quantify algorithmic bias in models. 

## 3.Datasets <a name="Datasets"></a>
Evaluation and mitigation of biases is applied to two datasets:

1. [Taiwanese Credit Card Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
This dataset comprises of customers' default payments in Taiwan in 2005.
   **Some features of the data:**
   
   Target Variable: *Default/Non-Default*
   
   Faetures: *23*
   
   Instances: *30000*


2. [Adult(Census) Data set](https://archive.ics.uci.edu/dataset/2/adult)
This dataset comprisesof an individual’s annual income results from various factors. Also known as "Census Income" dataset.

   **Some features of the data:**
   
   Target Variable: *Income	>50K, <=50K*
   
   Faetures: *14*
   
   Instances: *48842*



## 4.Overall Methodology <a name="Methodology"></a>
![Picture1](https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/assets/112579333/f496d2a1-747a-48dd-a458-6628174149b9)




### 4.1 Building an unmitigated model

We opted for a baseline model using XGBoost, selected to align with client stipulations and preferences. XGBoost is preferred by clients due to its scalability and its enhanced performance capabilities. 

Additionally, feature importance plots help make the model somewhat explainable. 


### 4.2 Fit Python Packages

Using various packages, we tried to mitigate the bias in the baseline model. We rigorously assess these fairness tools to ensure their reliability and effectiveness, allowing us to offer informed recommendations on their practical use. We try various methods and metrics for each package to evaluate their effectiveness and compatibility with the client’s needs. 

The 10 Python packages we explored are:

Package Name, GitHub Repository

1. AI Fairness 360, https://github.com/Trusted-AI/AIF360
2. DALEX, https://github.com/ModelOriented/DALEX
3. Deon, https://github.com/drivendataorg/deon
4. Fairlearn, https://github.com/fairlearn/fairlearn
5. fairness-in-ml, https://github.com/equialgo/fairness-in-ml
6. FairSight, https://github.com/ayong8/FairSight
7. Responsible AI toolbox, https://github.com/microsoft/responsible-ai-toolbox
8. Smclarify, https://github.com/aws/amazon-sagemaker-clarify
9. Themis-ML, https://github.com/cosmicBboy/themis-ml
10. PiML, https://github.com/SelfExplainML/PiML-Toolbox


Out of the ten packages, AI Fairness 360 outperforms the rest in terms of bias mitigation, usability, generalizability. 
# include details about aifairness 360 and the techniques used
**AIFairness 360** 





Below, we include a brief description of the other 9 packages that we tested for bias mitigation and teh reasons behind their exclusion from the experimental analysis.








**DALEX** 

DALEX is designed to work with any predictive model, regardless of its internal structure. It offers various tools for exploring different aspects of a model, including model performance, conditional effects of variables, and variable importance. This enables a deeper understanding of how models make their predictions. However, the key limitation of this package is that it serves only as a starting point for understanding a model but does not include any tools to pinpoint model fairness deficiencies and mitigate the unfairness. Because of this limitation, we consider DALEX unfit for the purpose of bias mitigation proposed by our clients and removed it from consideration. 


**Deon**

Deon is a command-line tool that appends an ethics checklist to the project for data scientists to assess biases manually. Even though the checklist has a comprehensive list of criteria to evaluate bias, the package so solely based on data scientists’ subjective opinions on evaluating biases. There is no algorithm to calculate bias and perform mitigation. Thus, the Deon package was removed from our consideration. 


**Fairlearn**

Fairlearn is an open-source library designed to help researchers assess and mitigate unfairness in machine learning models. It provides tools for evaluating and visualizing fairness metrics and helps users understand and address potential biases in their models. In addition to assessment, Fairlearn provides algorithms to mitigate unfairness in machine learning models. These algorithms aim to adjust model predictions to reduce disparities while maintaining overall predictive performance. There are two different mitigation strategies in the Fairlearn package, Postprocessing and Reductions. We have decided to remove Fairlearn for consideration because one of the only two mitigation methods, “Reductions”, failed to reduce fairness for our dataset. We speculated that this may be due to the reason that the “Reductions” approach aims to reduce biases from multiple features holistically. Thus, in our analysis when we only focus on one feature at a time, the mitigation was not prioritizing the feature we selected. Since the inner workings of which feature the “Reductions” approach prioritizes was missing from the documentation, we also find this package relatively difficult to work with. 


**Themis-ML**

Themis-ML is designed to promote fairness-aware machine learning. It builds upon pandas and scikit-learn, implementing various algorithms that address discrimination and bias in machine learning models. However, many mitigation methods are missing documentation and explanations, making the package difficult to use. Thus, we have decided to move on with other packages. 

**PiML**

PiML is a toolbox for interpretable machine learning model development and validation. It helps users to use simple codes to build ML models and interpret their performance results. Within the results section, fairness was evaluated, under the integration of another fairness package called solas-ai. The PiML package itself was not a package aimed at bias evaluation and mitigation. Thus, we don’t think PiML is useful for our clients’ purposes. As for the fairness package solas-ai, it is useful for disparity testing and bias evaluation, but it has no fairness mitigation methods. 

**Responsible AI toolbox**

The Responsible AI Toolbox encompasses a comprehensive array of tools, including libraries and user interfaces, aimed at enhancing the scrutiny and assessment of data and models. This toolkit is instrumental in enriching the understanding of AI systems, providing those involved in AI development and oversight with the necessary resources to foster ethical and responsible AI practices. It supports informed decision-making based on data, ensuring AI technologies are developed and managed with greater responsibility. However, it falls short of offering a tool for mitigation.

**Smclarify**

The smclarify package, provided by Amazon Web Services (AWS), is a powerful tool within the AWS SageMaker suite designed for bias detection and explainability in machine learning models. It aids in identifying and reporting various types of biases in datasets and models—both pre- and post-training—to promote fairness. Additionally, it offers explainability features to elucidate the impact of input features on model predictions, which is vital for model transparency, debugging, and improvement. However, It focuses on identifying and explaining bias rather than providing direct solutions or strategies to mitigate these biases within machine learning models.

**fairness-in-ml**

Fairness in ML mimics the Generative Adversarial Network logic of a zero-sum game, where the generative model is replaced by the predictive classifier model and the task of the adversarial model is to predict the sensitive attribute value from the output of the classifier. The adversarial training of the classifier is done through the extension of the original network architecture with an adversarial component. This technique ranks low in terms of Generalizability, Useability, and Interpretation as it involves the architecture of neural networks which is harder to implement and interpret when compared to XGBoost model. Therefore, we also decided against moving this package forward in our recommendation.



### 4.3 Performance Analysis and Comparison
#### 4.3.1 Fairness Evaluation Metrics
There has been a wealth of studies on fairness in machine learning and algorithmic biases in recent years. Specifically, scholars have proposed several definitions of fairness and different metrics that quantify bias – such as statistical parity, equalized odds, and disparate impact.
Our analysis will cover a suite of fairness metrics but will focus on Disparate Impact. We zero in on the disparate impact metric due to its salience in the consumer lending space. Disparate Impact measures the ratio between the proportion of each group receiving the positive outcome. This is a commonly cited metric measuring fairness in financial decisions. In fact, the Consumer Compliance handbook published by the Board of Governors at the Federal Reserve highlights disparate impact as a textbook example of a violation of the ECOA.

The Disparate Impact (DI) is calculated using the formula:

![1_yeR8SOoMQX82OZirmDlA-A](https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/assets/112579333/159fe840-ba8c-417e-93b6-5e8a2068a0a6)




#### 4.3.2 Balanced Accuracy 
Balanced Accuracy calculates the mean between the True Positive Rate and the True Negative Rate in the model predictions. We want our clients to be able to grant financial products to people who would not default and deny them to people who would. Balanecd Accuracy is also ideal for Unbalanced Data, in cases where there are very few defaults in the data, this metric can capture model performance accurately.





## 5.Results <a name="Results"></a>


## include results from matching 



The pre and post processing techniques of AIF 360 on mitigation of biases across race as the sensitive attribute on the adult dataset yield the following results.We can significant reduction in disparate impacts in many of these techniques while maintaining manageable degradation in performance. 

![output](https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/assets/112579333/d13c2383-2db9-46fd-ba28-c10c1071fc85)



## 6.Conclusion <a name="Conclusion"></a>
In conclusion of the analysis oerformed, we recommend to follow a fairness pipeline as shown:


<img width="1430" alt="Screenshot 2024-04-03 at 8 06 12 PM" src="https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/assets/112579333/f8c1fba7-2f07-492a-beb4-3edee2555bf8">


## 7.Usage <a name="Usage"></a>


```
# Create a virutal env to install the requierd packages
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate

# Clone the repository and install the required packages
git clone https://github.com/YYLinn/2nd-order-solution-ML-Fairness-.git
cd 2nd-order-solution-ML-Fairness
pip install -r requirements.txt

# Run the analysis script to perform the required experimentation
python 03_analysis/Experimentation/main.py --technique <<Fairness_technique>> --sensitive_attr <<sensitive attribute>>
```

The code will give json files with various performance metics and disparate impact of the new model, according to the technique specified.

Documentation related to the project can be found at: 
[Documentation](https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/tree/main/05_documents)


