# Fairness in Machine Learning for 2nd Order Solutions

#### Team Members: [Yiyu Lin](https://github.com/YYLinn), [Yinting Zhong](https://github.com/YintingZhong), [Genesis Qu](https://github.com/qu-genesis), [Pragya Raghuvanshi](https://github.com/pr-124))

## Introduction


## Dataset

**Some features of the data:**


**Overall Methodology**


![Picture1](https://github.com/YYLinn/2nd-order-solution-ML-Fairness-/assets/112579333/f496d2a1-747a-48dd-a458-6628174149b9)

1. Building an unmitigated model

2. Fit Python packages (Fairness Assessment+Bias Mitigation)

3. Performance Analysis and Comparison


**1. Building an unmitigated model**

We opted for a baseline model using XGBoost, selected to align with client stipulations and preferences. XGBoost is preferred by clients due to its scalability and its enhanced performance capabilities. 

Additionally, feature importance plots help make the model somewhat explainable. 


**2. Fit Python Packages**

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


Out of the ten packages, AI Fairness 360 performs the best…..


Below, we include a brief description of the other 9 packages and why we removed them from consideration.


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


