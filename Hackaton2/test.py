# %% [markdown]
# # **[LEPL1109] - STATISTICS AND DATA SCIENCES**
# ## **Hackaton 02 - Classification: Diabetes Health indicators**
# \
# Prof. D. Hainaut\
# Prod. L. Jacques\
# \
# \
# Adrien Banse (adrien.banse@uclouvain.be)\
# Jana Jovcheva (jana.jovcheva@uclouvain.be)\
# François Lessage (francois.lessage@uclouvain.be)\
# Sofiane Tanji (sofiane.tanji@uclouvain.be)

# %% [markdown]
# ![alt text](figures/diab_illustration.jpg)

# %% [markdown]
# <div class="alert alert-danger">
# <b>[IMPORTANT] Read all the documentation</b>  <br>
#     Make sure that you read the whole notebook, <b>and</b> the <code>README.md</code> file in the folder.
# </div>

# %% [markdown]
# # **Guidelines and Deliverables**
# 
# *   This hackaton is due on the **29 November 2024 at 23h59**
# *   Copying code or answers from other groups (or from the internet) is strictly forbidden. <b>Each source of inspiration (stack overflow, git, other groups, ChatGPT...) must be clearly indicated!</b>
# *  This notebook (with the "ipynb" extension) file, the Python source file (".py"), the report (PDF format) and all other files that are necessary to run your code must be delivered on <b>Moodle</b>.
# * Only the PDF report and the python source file will be graded, both on their content and the quality of the text / figures.
#   * 4/10 for the code.
#   * 4/10 for the Latex report.
#   * 2/10 for the vizualisation. <br><br>
# 
# <div class="alert alert-info">
# <b>[DELIVERABLE] Summary</b>  <br>
# After the reading of this document (and playing with the code!), we expect you to provide us with:
# <ol>
#    <li> a PDF file (written in LaTeX, see example on Moodle) that answers all the questions below. The report should contain high quality figures with named axes (we recommend saving plots with the <samp>.pdf</samp> extension);
#    <li> a Python file with your classifier implementation. Please follow the template that is provided and ensure it passes the so-called <i>sanity</i> tests;
#    <li> this Jupyter Notebook (it will not be read, just checked for plagiarism);
#    <li> and all other files (not the datasets!) we would need to run your code.
# </ol>
# </div>
# 
# As mentioned above, plagiarism is forbidden. However, we cannot forbid you to use artificial intelligence BUT we remind you that the aim of this project is to learn classification on your own and with the help of the course material. Finally, we remind you that for the same question, artificial intelligence presents similar solutions, which could be perceived as a form of plagiarism.

# %% [markdown]
# # **Context & Objective**
# Diabetes is among the most prevalent chronic diseases in the United States, impacting millions of Americans each year and exerting a significant financial burden on the economy. Diabetes is a serious chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood, and can lead to reduced quality of life and life expectancy. After different foods are broken down into sugars during digestion, the sugars are then released into the bloodstream. This signals the pancreas to release insulin. Insulin helps enable cells within the body to use those sugars in the bloodstream for energy. Diabetes is generally characterized by either the body not making enough insulin or being unable to use the insulin that is made as effectively as needed.\
# Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes. While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients. Early diagnosis can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.
# 
# You work in the diabetology department at **Saint Luc University Hospital**. The head of the department has asked you to find a solution for classifying and predicting **whether patients are at high risk of developing diabetes**. This will enable them to schedule an appointment with these patients to set up prevention tools. To do this, you have a database of patients who have passed through the department in recent years. In addition, the head of the department feels that the poll is too long, and would like to **reduce the number of questions while maintaining the reliability and quality of the results**.\
# Your aim is to determine which characteristics are relevant and enable reliable patient classification. Be careful, don’t let a potential diabetic patient slip through the cracks. The rest of this document will guide you in this process.
# 
# ## **Dataset description**
# 
#  
# The data set is a real-world data set based on a survey (BRFSS) conducted by the Centers for Disease Control and Prevention in the USA some ten years ago.\
# The Behavioral Risk Factor Surveillance System (BRFSS) is an annual telephone health survey conducted by the Centers for Disease Control and Prevention. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic diseases and use of preventive services. The survey has been conducted annually since 1984. It contains 22 headings and around 70,000 entries.
# 
# 
# <img src="figures/Features_table.png" alt="drawing" width="800"/>
# 
# ## **Notebook structure**
# 
# * PART 1 - Preliminaries
#    - 1.1 - Importing the packages
#    - 1.2 - Importing the dataset
#    - 1.3 - Is the dataset balanced?
#    - 1.4 - Scale the dataset
#     <br><br>
# * PART 2 - Correlation
#    - 2.1 - Correlation matrix 
#    - 2.2 -Analyze the correlation with diabetes
#    - 2.3 - Model selection and parameters tuning
#    - 2.4 - Precision-Recall curve and thresholding
#    <br><br>
# * PART 3 - Classifiers
#    - 3.1 - Linear regressor
#    - 3.2 - Logisitic regressor
#    - 3.3 - KNN regressor
#    <br><br>
# * PART 4 - Validation metrics
#    - 4.1 - Precision score
#    - 4.2 - Recall score
#    - 4.3 - F1 score
#    <br><br>
# * PART 5 - Reduce the questionnaire size
#    - 5.1 - K-Fold preparation
#    - 5.2 - Find the right combination length/regressor
#    - 5.3 - Visualize the scores
#    <br><br>   
# * PART 6 - Visualization
#    - 6.1 - Visualize your results
# 
# We filled this notebook with preliminary (trivial) code. This practice makes possible to run each cell, even the last ones, without throwing warnings. <b>Take advantage of this aspect to divide the work between all team members!</b> <br><br>
# Remember that many libraries exist in Python, so many functions have already been developed. Read the documentation and don't reinvent the wheel! You can import whatever you want.
# 

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART I - Preliminaries</b> </font> <br><br>

# %% [markdown]
# In this part of the hackathon, we will import the necessary packages, then we will import the dataset, scale it and analyze its distribution.

# %%
"""
CELL N°1.1 : IMPORTING ALL THE NECESSARY PACKAGES

@pre:  /
@post: The necessary packages should be loaded.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
# Import all the necessary packages here...

# %%
"""
CELL N°1.2 : IMPORTING THE DATASET

@pre:  /
@post: The object `df` should contain a Pandas DataFrame corresponding to the file `diabetes_dataset.csv`
"""

file = "diabetes_dataset.csv"
df = pd.read_csv(file)

df = pd.DataFrame(df) # To modify
df
# df.describe()
# df.info()


# %% [markdown]
# ***Is the dataset balanced?***
# 
# It's good practice to check this to better understand the contents of our dataset. The balance between the different classes has an impact on the binarization threshold (which is initialized here at 0.5). Other things can also have an impact on the choice of threshold.

# %%
"""
CELL N°1.3 : IS THE DATASET BALANCED?

@pre:  `df` contains the dataset
@post: Plot the diabetic/non-diabetic distribution in a pie chart
"""

# Plot the diabetic/non-diabetic distribution in a pie chart here...

df['Diabetes'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# Here! 

# %% [markdown]
# ***Standardize*** is important when you work with data because it allows data to be compared with one another. 
# 
# $z$ is the standard score of a population $x$. It can be computed as follows:
# $$z = \frac{x-\mu}{\sigma}$$
# with $\mu$ the mean of the population and $\sigma$ the standard deviation of the poplutation.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Standard_score) for further information about the standardization.\
# Be careful to use the same formula as us, check in `scikit-learn`
# 

# %%
"""
CELL N°1.4 : SCALE THE DATASET

@pre:  A pandas.DataFrame `df` containing the dataset
@post: A pandas.DataFrame `df` containing the standardized dataset (except classification columns (Diabetes))

"""
from sklearn import preprocessing
def scale_dataset(df): 
    # Modify here...
    columns = df.iloc[:, 1:].columns
    df[columns] = preprocessing.scale(df[columns])

    return df

df = scale_dataset(df)
df
# df.info()
# df.describe()

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART II - Correlation</b> </font> <br><br>

# %% [markdown]
# ***In order to keep*** the important features for our classification, we can compute and plot (see e.g. `seaborn.heatmap`) the correlation matrix. With these correlation coefficient, we can establish a feature selection strategy.\
# Be sure to use the `pearson` correlation.
# 

# %%
import seaborn
"""
CELL N°2.1 : CORRELATION MATRIX

@pre:  `df` contains the diabetes dataset
@post: `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the full dataset
"""

corr_matrix = df.corr(method='pearson')
seaborn.heatmap(corr_matrix,cmap='coolwarm',vmin=-1,vmax=1)

# %% [markdown]
# After this visualization, it is time to sort the coefficients of correlation to keep them with the best correlation with `Diabetes`. **Be careful** with the sign.

# %%
"""
CELL N°2.2 : ANALYZE THE CORRELATION WITH DIABETE

@pre:  `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the training set
@post: `sorted_features` contains a list of features (columns of `df`) 
       sorted according to their correlation with `Diabetes` 
"""

def sort_features(corr_matrix):
    corr = corr_matrix["Diabetes"].drop("Diabetes")
    return corr.sort_values(key=abs,ascending=False).axes[0].to_list()

sorted_features = sort_features(corr_matrix)
print(sorted_features)

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART III - Classifiers</b> </font> <br><br>

# %% [markdown]
# In this third part, you need to write functions that return a lamba function with a classifier for the test set. **Be careful** to keep the same form as the one suggested to pass the sanity checks.

# %% [markdown]
# **Implement** the *linear_regressor*. Please follow the specifications in the provided template.
# 
# **Reminder:** Linear regressor is a model that predicts a continuous value by fitting a line (or hyperplane) to the data, minimizing the difference between observed and predicted values.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) for further information about the classifier.

# %%
"""
CELL N°3.1 : LINEAR REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

from sklearn.linear_model import LinearRegression

def linear_regressor(X_train, y_train, threshold = 0.5):
    # To modify

    model = LinearRegression()
    model.fit(X_train, y_train)
    return lambda X_test: (model.predict(X_test) > threshold).astype(int)

# %% [markdown]
# **Implement** the *logistic_regressor*. Please follow the specifications in the provided template.
# 
# **Reminder:** Logisitic regressor is a classification model that estimates the probability of an observation belonging to a class using a logistic function; suitable for binary (and multiclass) problems.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) for further information about the classifier.

# %%
"""
CELL N°3.2 : LOGISTIC REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post:  Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

from sklearn.linear_model import LogisticRegression

def logistic_regressor(X_train, y_train, threshold = 0.5):
    # To modify
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return lambda X_test: (model.predict(X_test) > threshold).astype(int)

# %% [markdown]
# **Implement** the *knn_regressor*. Please follow the specifications in the provided template.  <br>
# 
# **Reminder:** Knn regressor is a non-parametric classification algorithm that classifies an observation according to the classes of its k nearest neighbors in feature space.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for further information about the classifier.\
# Attention, you must implement it with **Euclidian distance** and **10** neighbors.

# %%
"""
CELL N°3.3 : KNN REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

from sklearn.neighbors import KNeighborsClassifier

def knn_regressor(X_train, y_train, threshold = 0.5, n_neighbors = 10):
    # To modify
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X_train, y_train)
    return lambda X_test: model.predict(X_test)

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART IV - Validation metrics</b> </font> <br><br>

# %% [markdown]
# In this part, we will implement tools that will help us to **validate** the prediction models implemented in Part III. In particular, we will use the _precision, recall_ and _F1 score_ metrics. 
# 
# **Implement** the _precision, recall_ and _F1 score_. Please follow the specifications in the provided template.  <br>

# %% [markdown]
# **Reminder**
# 
# $F_1$ is a performance metric allowing to obtain some trade-off between the precision and recall criterions. It can be computed as follows:
# $$F_1 = 2~\frac{\mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}.$$
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/F-score) for further information about the three metrics.

# %%
"""
CELL N°4.1 : PRECISION SCORE

@pre:  /
@post: `precision(y_test, y_pred)` returns the prediction metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""
from sklearn.metrics import precision_score

def precision(y_test, y_pred):
    
    return precision_score(y_test,y_pred)

# %%
"""
CELL N°4.2 : RECALL SCORE

@pre:  /
@post: `recall(y_test, y_pred)` returns the recall metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""
from sklearn.metrics import recall_score

def recall(y_test, y_pred):
    
    return recall_score(y_test,y_pred)

# %%
"""
CELL N°4.3 : F1 SCORE

@pre:  /
@post: `f1_score(y_test, y_pred)` returns the F1 score metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""
from sklearn.metrics import f1_score as f1
def f1_score(y_test, y_pred):
    
    return f1(y_test,y_pred)

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART V - Reduce the questionnaire size</b> </font> <br><br>

# %% [markdown]
# In this part, find a model that satisfies the following specifications: 
# - A recall of at least 95%
# - A F1 score of at least 75%
# 
# For that, we will use **k-fold** cross validation (see [the Wikipedia page](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) for a reminder), and then test the three models above with 
# - Different number of features
# - Different thresholds

# %% [markdown]
# In order to use k-fold cross validation, use the class `sklearn.model_selection.
# old` from the `scikit-learn` library (see [the documentation](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.KFold.html)) with `n_splits = 3`.
# 
# <div class="alert alert-danger">
# <b>[IMPORTANT] Grading</b>  <br>
# In order for us to be able to automatically grade your submission, put <code>shuffle=True</code>, and <code>random_state=1109</code> when you initialize <code>KFold</code>.
# </div>

# %%
"""
CELL N°5.1 : K-FOLD PREPARATION

@pre:  `df` contains the scaled dataset.
@post: The following specifications should be satisfied: 
            - `kf` should contain a `KFold` object with 3 splits, shuffled and with 1109 seed. 
            - `X` should contain a pd.DataFrame with all the features (all columns except `Diabetes`)
            - `y` should contain a pd.DataFrame with all the labels (only the column `Diabetes`)
"""


# Initialize KFold with 3 splits, shuffle=True, and random_state=1109
kf = KFold(n_splits=3, shuffle=True, random_state=1109)

# Separate the features (X) and labels (y) from the dataset
X = df.drop(columns=["Diabetes"])  # All columns except 'Diabetes' are features
y = df["Diabetes"]  # The 'Diabetes' column is the label

# %% [markdown]
# 

# %% [markdown]
# In order to find our model, proceed as follows: 
# - Fix a threshold in $(0, 1)$
# - Define a dictionary `result` of the form 
# 
# <code>result = {
#     "linear": {}, 
#     "logistic": {}, 
#     "knn": {}
# }
# </code>
# 
# - For $ i \in \{1, \dots, \texttt{N\_features}\} $: 
#     - Select the $i$ **most correlated features** (use `sorted_features` defined above)
#     - For all the pairs $((X_{\text{train}}, y_{\text{train}}), (X_{\text{test}}, y_{\text{test}}))$ given by k-fold
#         - Compute the linear, logistic and KNN regressors with the fixed threshold on $X_{\text{train}}$
#         - Compute the 3 different 3-tuple `validation(regressor, X_test, y_test)`
#     - In `result[reg][i]`, save the **average** of all the validation tuples you computed for `reg`

# %%
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

def validation(regressor, X_test, y_test):
    # Nothing to do here!
    y_pred = regressor(X_test)
    return (
        recall_score(y_test, y_pred), 
        precision_score(y_test, y_pred), 
        f1_score(y_test, y_pred)
    )

# Fix the threshold
threshold = 0.29

# Initialize the result dictionary
result = {
    "linear": {}, 
    "logistic": {}, 
    "knn": {}
}

# Assuming `sorted_features` contains the features sorted by correlation
# This variable should be defined before using it
N_features = len(X.columns)

# Iterate over the number of features to keep
for i in range(1, N_features + 1):
    # Select the `i` most correlated features
    selected_features = sorted_features[:i]
    
    # Initialize lists to store validation metrics for each regressor
    linear_metrics = []
    logistic_metrics = []
    knn_metrics = []
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
        # Split the dataset
        X_train, X_test = X.iloc[train_index][selected_features], X.iloc[test_index][selected_features]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Convert to NumPy arrays
        X_train_np, X_test_np = X_train.values, X_test.values
        y_train_np, y_test_np = y_train.values, y_test.values
        
        # Linear regression
        linear_pred = linear_regressor(X_train_np, y_train_np, threshold)(X_test_np)
        linear_metrics.append(validation(lambda X: linear_pred, X_test_np, y_test_np))
        
        # Logistic regression
        logistic_pred = logistic_regressor(X_train_np, y_train_np, threshold)(X_test_np)
        logistic_metrics.append(validation(lambda X: logistic_pred, X_test_np, y_test_np))
        
        # KNN
        knn_pred = knn_regressor(X_train_np, y_train_np, threshold)(X_test_np)
        knn_metrics.append(validation(lambda X: knn_pred, X_test_np, y_test_np))
    
    # Compute the average of the validation metrics for each regressor
    result["linear"][i] = tuple(np.mean(metric) for metric in zip(*linear_metrics))
    result["logistic"][i] = tuple(np.mean(metric) for metric in zip(*logistic_metrics))
    result["knn"][i] = tuple(np.mean(metric) for metric in zip(*knn_metrics))

# `result` now contains the average metrics for all combinations of regressors and feature counts


# %% [markdown]
# The following cell allows you to test if the threshold that you chose satisfies the specifications, that are 
# - A recall of at least 95%
# - A F1 score of at least 75%
# 
# Plot these graphs for different threshold, and select the model **with the smallest number of questions** that satisfy the conditions above.

# %%
"""
CELL N°5.3 : VISUALIZE THE SCORES

@pre:  `result` contains the average of the validations for regressor `reg`, when keeping the `i` most correlated features
@post: plot of the scores for each condition
"""

# Nothing to do here, just run me! 

from helper import plot_result
plot_result(result, threshold, to_show = "recall")
plot_result(result, threshold, to_show = "f1_score")
print(result)

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART VI - Visualization</b> </font> <br><br>
# 

# %% [markdown]
# In this part, you are asked to produce a **clear and clean figure** expressing a result
# or giving an overall vision of your work for this hackaton. **Please feel free to do as you
# wish. Be original!** 
# 
# The **clarity**, **content** and **description** (in the report) of your figure will be evaluated.

# %%
import ipywidgets as widgets
from IPython.display import display
s = sorted_features[:5]
print(s)
df

# Création d'un widget de sélection
choix1 = widgets.ToggleButtons(
    options=['oui', 'non'],
    description=s[0],
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' ou '' (vide)
)
choix2 = widgets.ToggleButtons(
    options=['oui', 'non'],
    description=s[1],
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' ou '' (vide)
)
choix3 = widgets.ToggleButtons(
    options=['oui', 'non'],
    description=s[2],
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' ou '' (vide)
)
choix4 = widgets.ToggleButtons(
    options=['oui', 'non'],
    description=s[3],
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' ou '' (vide)
)
choix5 = widgets.ToggleButtons(
    options=['oui', 'non'],
    description=s[4],
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' ou '' (vide)
)

# Afficher le widget
display(choix1)
display(choix2)
display(choix3)
display(choix4)
display(choix5)

# Réutilisation de la valeur
def enregistrer_reponse(change):
    print(f"Vous avez choisi : {choix1.value}")
    # Ajoute ici du code pour stocker ou utiliser la réponse

choix1.observe(enregistrer_reponse, names='value')
choix2.observe(enregistrer_reponse, names='value')
choix3.observe(enregistrer_reponse, names='value')
choix4.observe(enregistrer_reponse, names='value')
choix5.observe(enregistrer_reponse, names='value')




# %%


# %%



