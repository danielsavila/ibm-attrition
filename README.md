<!-- ABOUT THIS HR ATTRITION PROJECT -->
# About The IBM-Attrition Project

This project was born out of an interest in analyzing and producing an attrition classifier for human analytics. While at the University of Chicago, I worked with hospital data that indicated significant quality of life improvements could be made by reducing work-related burnout for doctors, in particular hospitalists. Reducing burnout had potential to lead to better quality of life outcomes for doctors, better outcomes healthcare for patients, better quality of life for patients while hospitalized, and improvements toward hospital profitability.

This project takes Kaggle data (https://www.kaggle.com/datasets/rohitsahoo/employee) regarding employee attrition at IBM and attempts to fit a model that optimizes attrition predictions using recall as the primary evaluation metric over a given feature set. 

## Why is predicting attrition important:
* McKinsey estimates that the average cost of replacing a full time employee is $52,000, which incentivizes firms to get the hiring process right the first time. (https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/increasing-your-return-on-talent-the-moves-and-metrics-that-matter)
* While AI has eliminated numerous entry-level job openings, the opportunity cost of hiring the WRONG junior talent has increased as recruiters now need to filter through an exponentially increasing number of applicants.
* While the supply of inexperineced junior talent baloons, competition for experienced senior talent has increased as well, leading to higher pay packages, increased hiring timelines, and the potential for poaching from other firms.

If there was...
1) a way to identify individuals who were at high risk of leaving the firm, and 
2) a low cost intervention that increased the probability of that employee staying, then

there exists a break even point where the total cost of implementation and the total net savings are equal. Therefore, any implementation with total cost below that threshold would contribute net savings for the company. 

### i.e. if we can create a targeted retention program that, on average, costs less than losing an employee, we can save the firm some money, save years of intangible experience, and potentially maintain/improve company morale.

## About this repo:
This repo contains 5 .py files that contain different models fit to the data. The models are the following...
* K-Nearest Neighbors
* Linear/Quadratic Discriminant Analysis
* Logistic Regression
* Support Vector Machines (linear, polynomial, radial kernels)
* XGBoostClassifier


### Built With

This project was built using the following libraries...
* Pandas
* Numpy
* Scikit-Learn
* DataBricks
* Imbalanced-Learn
* MatPlotLib
* Seaborn
* XGBoost

and the following data science techniques...

--  Preprocessing
* Feature Scaling
* Random Oversampling
* Synthetic Data Generation (SMOTE)

--  Model Validation/Evaluation
* Cross Validation
* Confusion Matrices
* Hyperparameter Tuning (GridSearch)
* Evaluation Metrics (recall, f-1)
* Feature Importance


<!-- CONTACT -->
## Contact

Daniel Avila - (https://www.linkedin.com/in/daniel-avila-123392149/) - danielsavila2020@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
