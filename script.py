# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:57.178124Z","iopub.execute_input":"2026-02-14T15:10:57.178523Z","iopub.status.idle":"2026-02-14T15:10:57.536700Z","shell.execute_reply.started":"2026-02-14T15:10:57.178491Z","shell.execute_reply":"2026-02-14T15:10:57.535807Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Employee Attrition Risk Analysis – Salifort Motors
# 
# ## Project Overview
# Employee turnover can significantly impact operational costs, productivity, and
# organizational stability. This project analyzes employee data from Salifort Motors
# to identify factors associated with employee attrition and to build a predictive
# model that identifies employees at risk of leaving.
# 
# ## Objectives
# - Explore workforce characteristics and attrition patterns
# - Identify key drivers of employee turnover
# - Build a predictive model for attrition risk
# - Provide actionable HR recommendations
# 
# ## Tools
# Python, pandas, numpy, matplotlib, seaborn, scikit-learn

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:57.538125Z","iopub.execute_input":"2026-02-14T15:10:57.538694Z","iopub.status.idle":"2026-02-14T15:10:59.594426Z","shell.execute_reply.started":"2026-02-14T15:10:57.538649Z","shell.execute_reply":"2026-02-14T15:10:59.593158Z"}}
# Import packages

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import warnings
warnings.filterwarnings("ignore")

# For saving models
import pickle

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.595926Z","iopub.execute_input":"2026-02-14T15:10:59.596624Z","iopub.status.idle":"2026-02-14T15:10:59.660311Z","shell.execute_reply.started":"2026-02-14T15:10:59.596596Z","shell.execute_reply":"2026-02-14T15:10:59.659570Z"}}
# RUN THIS CELL TO IMPORT YOUR DATA. 

# Load dataset into a dataframe

df = pd.read_csv('/kaggle/input/salifort-motors-employee-attrition/HR_capstone_dataset.csv')

# Display first few rows of the dataframe

df.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.662335Z","iopub.execute_input":"2026-02-14T15:10:59.662664Z","iopub.status.idle":"2026-02-14T15:10:59.686977Z","shell.execute_reply.started":"2026-02-14T15:10:59.662641Z","shell.execute_reply":"2026-02-14T15:10:59.686212Z"}}
# Gather basic information about the data 
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.688076Z","iopub.execute_input":"2026-02-14T15:10:59.688443Z","iopub.status.idle":"2026-02-14T15:10:59.719553Z","shell.execute_reply.started":"2026-02-14T15:10:59.688418Z","shell.execute_reply":"2026-02-14T15:10:59.718803Z"}}
# Gather descriptive statistics about the data

df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.720679Z","iopub.execute_input":"2026-02-14T15:10:59.721088Z","iopub.status.idle":"2026-02-14T15:10:59.729691Z","shell.execute_reply.started":"2026-02-14T15:10:59.721051Z","shell.execute_reply":"2026-02-14T15:10:59.728940Z"}}
# Rename columns as needed

df = df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Display all column names after the update

df.columns

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.730916Z","iopub.execute_input":"2026-02-14T15:10:59.731639Z","iopub.status.idle":"2026-02-14T15:10:59.746249Z","shell.execute_reply.started":"2026-02-14T15:10:59.731613Z","shell.execute_reply":"2026-02-14T15:10:59.745272Z"}}
# Check for missing values
df.isna().sum()

# %% [markdown]
# We do not have any missing values in our Dataset

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.747382Z","iopub.execute_input":"2026-02-14T15:10:59.747712Z","iopub.status.idle":"2026-02-14T15:10:59.766503Z","shell.execute_reply.started":"2026-02-14T15:10:59.747687Z","shell.execute_reply":"2026-02-14T15:10:59.765607Z"}}
# Check for duplicates

df.duplicated().sum()

# %% [markdown]
# We have 3008 duplicates in our Dataset 
# 
# That's 20.1% of our data

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.767601Z","iopub.execute_input":"2026-02-14T15:10:59.768040Z","iopub.status.idle":"2026-02-14T15:10:59.786427Z","shell.execute_reply.started":"2026-02-14T15:10:59.768008Z","shell.execute_reply":"2026-02-14T15:10:59.785772Z"}}
# Inspect some rows containing duplicates as needed

df[df.duplicated()].head()

# %% [markdown]
# The above output shows the first five occurences of rows that are duplicated farther down in the dataframe. How likely is it that these are legitimate entries? In other words, how plausible is it that two employees self-reported the exact same response for every column?
# 
# We could perform a likelihood analysis by essentially applying Bayes' theorem and multiplying the probabilities of finding each value in each column, but this does not seem necessary. With several continuous variables across 10 columns, it seems very unlikely that these observations are legitimate. We will proceed by dropping them

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.789088Z","iopub.execute_input":"2026-02-14T15:10:59.789321Z","iopub.status.idle":"2026-02-14T15:10:59.808405Z","shell.execute_reply.started":"2026-02-14T15:10:59.789301Z","shell.execute_reply":"2026-02-14T15:10:59.807476Z"}}
# Drop duplicates and save the resulting dataframe in a new variable as needed
df = df.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:10:59.809651Z","iopub.execute_input":"2026-02-14T15:10:59.810002Z","iopub.status.idle":"2026-02-14T15:11:00.019587Z","shell.execute_reply.started":"2026-02-14T15:10:59.809969Z","shell.execute_reply":"2026-02-14T15:11:00.018881Z"}}
# Create a boxplot to visualize the distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df['tenure'])
plt.show()

# %% [markdown]
# The above boxplot shows outliers in the tenure variable.
# 
# We are going to investigate how many rows in the data contain outliers in the tenure column.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:00.020585Z","iopub.execute_input":"2026-02-14T15:11:00.020932Z","iopub.status.idle":"2026-02-14T15:11:00.030984Z","shell.execute_reply.started":"2026-02-14T15:11:00.020907Z","shell.execute_reply":"2026-02-14T15:11:00.030119Z"}}
# Determine the number of rows containing outliers 

# Compute the 25th percentile value in `tenure`
percentile25 = df['tenure'].quantile(0.25)

# Compute the 75th percentile value in `tenure`
percentile75 = df['tenure'].quantile(0.75)

# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `tenure`
outliers = df[(df['tenure'] > upper_limit) | (df['tenure'] < lower_limit)]

# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:00.032193Z","iopub.execute_input":"2026-02-14T15:11:00.032613Z","iopub.status.idle":"2026-02-14T15:11:00.047562Z","shell.execute_reply.started":"2026-02-14T15:11:00.032578Z","shell.execute_reply":"2026-02-14T15:11:00.046556Z"}}
# Get numbers of people who left vs. stayed

print(df['left'].value_counts())
print()

# Get percentages of people who left vs. stayed

print(df['left'].value_counts(normalize=True))

# %% [markdown]
# We examine variables that we are interested in, and create plots to visualize relationships between variables in the data.
# 
# We start by creating a stacked boxplot showing the average_monthly_hours distributions for number_project, comparing the distributions of employees who stayed versus those who left.
# 
# Box plots are very useful in visualizing distributions within data, but they can be deceiving without the context of how big the sample sizes that they represent are. So, you could also plot a stacked histogram to visualize the distribution of number_project for those who stayed and those who left.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:00.048908Z","iopub.execute_input":"2026-02-14T15:11:00.049386Z","iopub.status.idle":"2026-02-14T15:11:00.594094Z","shell.execute_reply.started":"2026-02-14T15:11:00.049360Z","shell.execute_reply":"2026-02-14T15:11:00.593243Z"}}
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df[df['left']==0]['number_project']
tenure_left = df[df['left']==1]['number_project']
sns.histplot(data=df, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()

# %% [markdown]
# It might be natural that people who work on more projects would also work longer hours. This appears to be the case here, with the mean hours of each group (stayed and left) increasing with the number of projects worked. However, a few things stand out from this plot.
# 
# There are two groups of employees who left the company: (A) those who worked considerably less than their peers with the same number of projects, and (B) those who worked much more. Of those in group A, it's possible that they were fired. It's also possible that this group includes employees who had already given their notice and were assigned fewer hours because they were already on their way out the door. For those in group B, it's reasonable to infer that they probably quit. The folks in group B likely contributed a lot to the projects they worked in; they might have been the largest contributors to their projects.
# 
# Everyone with seven projects left the company, and the interquartile ranges of this group and those who left with six projects was ~255–295 hours/month, much more than any other group.
# 
# The optimal number of projects for employees to work on seems to be 3–4. The ratio of left/stayed is very small for these cohorts.
# 
# If you assume a work week of 40 hours and two weeks of vacation per year, then the average number of working hours per month of employees working Monday–Friday = 50 weeks * 40 hours per week / 12 months = 166.67 hours per month. This means that, aside from the employees who worked on two projects, every group—even those who didn't leave the company—worked considerably more hours than this. It seems that employees here are overworked.
# 
# As the next step, we are going to confirm that all employees with seven projects left.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:00.595075Z","iopub.execute_input":"2026-02-14T15:11:00.595490Z","iopub.status.idle":"2026-02-14T15:11:00.603245Z","shell.execute_reply.started":"2026-02-14T15:11:00.595464Z","shell.execute_reply":"2026-02-14T15:11:00.602182Z"}}
# Get value counts of stayed/left for employees with 7 projects
df[df['number_project']==7]['left'].value_counts()

# %% [markdown]
# This confirms that all employees with 7 projects have left.
# 
# Next, we are going to examine the average monthly hours versus the satisfaction levels.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:00.604412Z","iopub.execute_input":"2026-02-14T15:11:00.604976Z","iopub.status.idle":"2026-02-14T15:11:01.595919Z","shell.execute_reply.started":"2026-02-14T15:11:00.604938Z","shell.execute_reply":"2026-02-14T15:11:01.594821Z"}}
# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'stayed', 'left'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

# %% [markdown]
# The scatterplot above shows that there was a sizeable group of employees who worked ~240–315 hours per month. 315 hours per month is over 75 hours per week for a whole year. It's likely this is related to their satisfaction levels being close to zero.
# 
# The plot also shows another group of people who left, those who had more normal working hours. Even so, their satisfaction was only around 0.4. It's difficult to speculate about why they might have left. It's possible they felt pressured to work more, considering so many of their peers worked more. And that pressure could have lowered their satisfaction levels.
# 
# Finally, there is a group who worked ~210–280 hours per month, and they had satisfaction levels ranging ~0.7–0.9.
# 
# Note the strange shape of the distributions here. This is indicative of data manipulation or synthetic data.
# 
# For the next visualization, it might be interesting to visualize satisfaction levels by tenure.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:01.597020Z","iopub.execute_input":"2026-02-14T15:11:01.597846Z","iopub.status.idle":"2026-02-14T15:11:02.279921Z","shell.execute_reply.started":"2026-02-14T15:11:01.597819Z","shell.execute_reply":"2026-02-14T15:11:02.278908Z"}}
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df[df['left']==0]['tenure']
tenure_left = df[df['left']==1]['tenure']
sns.histplot(data=df, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();

# %% [markdown]
# There are many observations we can make from this plot.
# 
# Employees who left fall into two general categories: dissatisfied employees with shorter tenures and very satisfied employees with medium-length tenures.
# Four-year employees who left seem to have an unusually low satisfaction level. It's worth investigating changes to company policy that might have affected people specifically at the four-year mark, if possible.
# The longest-tenured employees didn't leave. Their satisfaction levels aligned with those of newer employees who stayed.
# The histogram shows that there are relatively few longer-tenured employees. It's possible that they're the higher-ranking, higher-paid employees.
# As the next step in analyzing the data, you could calculate the mean and median satisfaction scores of employees who left and those who didn't.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:02.281110Z","iopub.execute_input":"2026-02-14T15:11:02.281453Z","iopub.status.idle":"2026-02-14T15:11:02.295423Z","shell.execute_reply.started":"2026-02-14T15:11:02.281428Z","shell.execute_reply":"2026-02-14T15:11:02.294424Z"}}
# Calculate mean and median satisfaction scores of employees who left and those who stayed
df.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])

# %% [markdown]
# As expected, the mean and median satisfaction scores of employees who left are lower than those of employees who stayed. Interestingly, among employees who stayed, the mean satisfaction score appears to be slightly below the median score. This indicates that satisfaction levels among those who stayed might be skewed to the left.
# 
# Next, we proceed to examine salary levels for different tenures.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:02.296708Z","iopub.execute_input":"2026-02-14T15:11:02.297025Z","iopub.status.idle":"2026-02-14T15:11:02.802971Z","shell.execute_reply.started":"2026-02-14T15:11:02.297002Z","shell.execute_reply":"2026-02-14T15:11:02.802011Z"}}
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Define short-tenured employees
tenure_short = df[df['tenure'] < 7]

# Define long-tenured employees
tenure_long = df[df['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');

# %% [markdown]
# The plots above show that long-tenured employees were not disproportionately comprised of higher-paid employees.
# 
# Next, we will explore whether there's a correlation between working long hours and receiving high evaluation scores. We are going to create a scatterplot of average_monthly_hours versus last_evaluation.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:02.804479Z","iopub.execute_input":"2026-02-14T15:11:02.805280Z","iopub.status.idle":"2026-02-14T15:11:03.864410Z","shell.execute_reply.started":"2026-02-14T15:11:02.805250Z","shell.execute_reply":"2026-02-14T15:11:03.863344Z"}}
# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

# %% [markdown]
# The following observations can be made from the scatterplot above:
# 
# The scatterplot indicates two groups of employees who left: overworked employees who performed very well and employees who worked slightly under the nominal monthly average of 166.67 hours with lower evaluation scores.
# There seems to be a correlation between hours worked and evaluation score.
# There isn't a high percentage of employees in the upper left quadrant of this plot; but working long hours doesn't guarantee a good evaluation score.
# Most of the employees in this company work well over 167 hours per month.
# Next, we are going to examine whether employees who worked very long hours were promoted in the last five years.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:03.865632Z","iopub.execute_input":"2026-02-14T15:11:03.866023Z","iopub.status.idle":"2026-02-14T15:11:04.732513Z","shell.execute_reply.started":"2026-02-14T15:11:03.865997Z","shell.execute_reply":"2026-02-14T15:11:04.731686Z"}}
# Create plot to examine the relationship between `average_monthly_hours` and `promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'stayed', 'left'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');

# %% [markdown]
# The plot above shows the following:
# 
# very few employees who were promoted in the last five years left
# 
# very few employees who worked the most hours were promoted
# 
# all of the employees who left were working the longest hours
# 
# Next, we inspect how the employees who left are distributed across departments.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:04.733498Z","iopub.execute_input":"2026-02-14T15:11:04.733788Z","iopub.status.idle":"2026-02-14T15:11:04.742325Z","shell.execute_reply.started":"2026-02-14T15:11:04.733764Z","shell.execute_reply":"2026-02-14T15:11:04.741418Z"}}
# Display counts for each department
df["department"].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:04.743454Z","iopub.execute_input":"2026-02-14T15:11:04.743807Z","iopub.status.idle":"2026-02-14T15:11:05.161505Z","shell.execute_reply.started":"2026-02-14T15:11:04.743774Z","shell.execute_reply":"2026-02-14T15:11:05.160696Z"}}
# Create stacked histogram to compare department distribution of employees who left to that of employees who didn't
plt.figure(figsize=(11,8))
sns.histplot(data=df, x='department', hue='left', discrete=1, 
             hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation=45)
plt.title('Counts of stayed/left by department', fontsize=14);

# %% [markdown]
# Across all the departments, there is no significant difference in the proportion of employees who left to those who stayed.
# 
# Lastly, you could check for strong correlations between variables in the data.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:05.162868Z","iopub.execute_input":"2026-02-14T15:11:05.163333Z","iopub.status.idle":"2026-02-14T15:11:05.622583Z","shell.execute_reply.started":"2026-02-14T15:11:05.163299Z","shell.execute_reply":"2026-02-14T15:11:05.621663Z"}}
# Plot a correlation heatmap
plt.figure(figsize=(16, 9))

#Remove non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

heatmap = sns.heatmap(numeric_df.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

# %% [markdown]
# The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

# %% [markdown]
# It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, too many projects, and generally lower satisfaction levels.
# 
# It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave.

# %% [markdown] {"execution":{"iopub.status.busy":"2026-02-12T03:52:57.674750Z","iopub.execute_input":"2026-02-12T03:52:57.675112Z","iopub.status.idle":"2026-02-12T03:52:57.682029Z","shell.execute_reply.started":"2026-02-12T03:52:57.675077Z","shell.execute_reply":"2026-02-12T03:52:57.680892Z"}}
# Next, we will: 
# 
# Determine which models are most appropriate
# 
# Construct the model
# 
# Confirm model assumptions
# 
# Evaluate model results to determine how well your model fits the data

# %% [markdown]
# ## Logistic regression
# Note that binomial logistic regression suits the task because it involves binary classification.
# 
# Before splitting the data, encode the non-numeric variables. There are two: department and salary.
# 
# department is a categorical variable, which means you can dummy it for modeling.
# 
# salary is categorical too, but it's ordinal. There's a hierarchy to the categories, so it's better not to dummy this column, but rather to convert the levels to numbers, 0–2.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:05.623658Z","iopub.execute_input":"2026-02-14T15:11:05.623947Z","iopub.status.idle":"2026-02-14T15:11:05.648645Z","shell.execute_reply.started":"2026-02-14T15:11:05.623925Z","shell.execute_reply":"2026-02-14T15:11:05.647979Z"}}
# Copy the dataframe
df_enc = df.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Display the new dataframe
df_enc.head()

# %% [markdown]
# We will create a heatmap to visualize how correlated variables are. Considering the variables we're interested in examining correlations between.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:05.649696Z","iopub.execute_input":"2026-02-14T15:11:05.650304Z","iopub.status.idle":"2026-02-14T15:11:05.881005Z","shell.execute_reply.started":"2026-02-14T15:11:05.650278Z","shell.execute_reply":"2026-02-14T15:11:05.880035Z"}}
# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()

# %% [markdown]
# We will proceed to create a stacked bar chart to visualize the number of employees across departments, comparing those who left with those who didn't.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:05.882994Z","iopub.execute_input":"2026-02-14T15:11:05.883281Z","iopub.status.idle":"2026-02-14T15:11:06.144369Z","shell.execute_reply.started":"2026-02-14T15:11:05.883255Z","shell.execute_reply":"2026-02-14T15:11:06.143497Z"}}
# Create a stacked bart plot to visualize number of employees across department, comparing those who left with those who didn't
# In the legend, 0 (purple color) represents employees who did not leave, 1 (red color) represents employees who left
pd.crosstab(df['department'], df['left']).plot(kind ='bar',color='mr')
plt.title('Counts of employees who left versus stayed across department')
plt.ylabel('Employee count')
plt.xlabel('Department')
plt.show()

# %% [markdown]
# Logistic regression is quite sensitive to outliers; it would be a good idea at this stage to remove the outliers in the tenure column that were identified earlier.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.149485Z","iopub.execute_input":"2026-02-14T15:11:06.149912Z","iopub.status.idle":"2026-02-14T15:11:06.170928Z","shell.execute_reply.started":"2026-02-14T15:11:06.149883Z","shell.execute_reply":"2026-02-14T15:11:06.169858Z"}}
# Select rows without outliers in `tenure` and save resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

# Display first few rows of new dataframe
df_logreg.head()

# %% [markdown]
# We isolate the outcome variable, which is the variable we want our model to predict

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.172265Z","iopub.execute_input":"2026-02-14T15:11:06.172625Z","iopub.status.idle":"2026-02-14T15:11:06.179935Z","shell.execute_reply.started":"2026-02-14T15:11:06.172590Z","shell.execute_reply":"2026-02-14T15:11:06.178985Z"}}
# Isolate the outcome variable
y = df_logreg['left']

# Display first few rows of the outcome variable
y.head() 

# %% [markdown]
# We proceed to select the features that we want to use in our model
# 
# The feature that will help us in achieving this is "left"

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.180920Z","iopub.execute_input":"2026-02-14T15:11:06.181293Z","iopub.status.idle":"2026-02-14T15:11:06.209671Z","shell.execute_reply.started":"2026-02-14T15:11:06.181238Z","shell.execute_reply":"2026-02-14T15:11:06.208921Z"}}
# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)

# Display the first few rows of the selected features 
X.head()

# %% [markdown]
# We are going to split the data into training set and testing set respectively. We won't forget to stratify based on the values in y, since the classes are unbalanced

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.210969Z","iopub.execute_input":"2026-02-14T15:11:06.211371Z","iopub.status.idle":"2026-02-14T15:11:06.225828Z","shell.execute_reply.started":"2026-02-14T15:11:06.211341Z","shell.execute_reply":"2026-02-14T15:11:06.224808Z"}}
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# %% [markdown]
# We construct a logistic regression model using the train dataset and fit into the test data

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.227343Z","iopub.execute_input":"2026-02-14T15:11:06.227686Z","iopub.status.idle":"2026-02-14T15:11:06.729563Z","shell.execute_reply.started":"2026-02-14T15:11:06.227638Z","shell.execute_reply":"2026-02-14T15:11:06.728890Z"}}
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

# %% [markdown]
# We test the logistic regression model. We are going to use this model to make predictions on the test data

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.730387Z","iopub.execute_input":"2026-02-14T15:11:06.730647Z","iopub.status.idle":"2026-02-14T15:11:06.737875Z","shell.execute_reply.started":"2026-02-14T15:11:06.730622Z","shell.execute_reply":"2026-02-14T15:11:06.737193Z"}}
# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)

# %% [markdown]
# We are going to create a confusion matrix to visualize the results of the logistic regression model.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.739156Z","iopub.execute_input":"2026-02-14T15:11:06.739618Z","iopub.status.idle":"2026-02-14T15:11:06.917694Z","shell.execute_reply.started":"2026-02-14T15:11:06.739589Z","shell.execute_reply":"2026-02-14T15:11:06.916736Z"}}
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()

# %% [markdown]
# The upper-left quadrant displays the number of true negatives. The upper-right quadrant displays the number of false positives. The bottom-left quadrant displays the number of false negatives. The bottom-right quadrant displays the number of true positives.
# 
# True negatives: The number of people who did not leave that the model accurately predicted did not leave.
# 
# False positives: The number of people who did not leave the model inaccurately predicted as leaving.
# 
# False negatives: The number of people who left that the model inaccurately predicted did not leave
# 
# True positives: The number of people who left the model accurately predicted as leaving
# 
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
# 
# We will create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.
# 
# we also will check the class balance in the data. In other words, check the value counts in the left column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.918957Z","iopub.execute_input":"2026-02-14T15:11:06.919433Z","iopub.status.idle":"2026-02-14T15:11:06.926643Z","shell.execute_reply.started":"2026-02-14T15:11:06.919397Z","shell.execute_reply":"2026-02-14T15:11:06.925969Z"}}
df_logreg['left'].value_counts(normalize=True)

# %% [markdown]
# There is an approximately 83%-17% split. So the data is not perfectly balanced, but it is not too imbalanced. If it was more severely imbalanced, we might want to resample the data to make it more balanced. In this case, we can use this data without modifying the class balance and continue evaluating the model.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.927726Z","iopub.execute_input":"2026-02-14T15:11:06.928085Z","iopub.status.idle":"2026-02-14T15:11:06.952247Z","shell.execute_reply.started":"2026-02-14T15:11:06.928062Z","shell.execute_reply":"2026-02-14T15:11:06.951424Z"}}
# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown]
# The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.

# %% [markdown]
# WE MOVE TO THE DECISION TREE MODEL
# 
# This approach covers implementation of Decision Tree and Random Forest.
# 
# Isolate the outcome variable.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.953349Z","iopub.execute_input":"2026-02-14T15:11:06.953690Z","iopub.status.idle":"2026-02-14T15:11:06.960687Z","shell.execute_reply.started":"2026-02-14T15:11:06.953657Z","shell.execute_reply":"2026-02-14T15:11:06.959958Z"}}
# Isolate the outcome variable
y = df_enc['left']

# Display the first few rows of `y`
y.head()

# %% [markdown]
# We proceed to select the features

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.961925Z","iopub.execute_input":"2026-02-14T15:11:06.962635Z","iopub.status.idle":"2026-02-14T15:11:06.989147Z","shell.execute_reply.started":"2026-02-14T15:11:06.962610Z","shell.execute_reply":"2026-02-14T15:11:06.988370Z"}}
# Select the features
X = df_enc.drop('left', axis=1)

# Display the first few rows of `X`
X.head()

# %% [markdown]
# We will split the data into training, validating and testing set

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:06.990070Z","iopub.execute_input":"2026-02-14T15:11:06.990490Z","iopub.status.idle":"2026-02-14T15:11:07.004924Z","shell.execute_reply.started":"2026-02-14T15:11:06.990457Z","shell.execute_reply":"2026-02-14T15:11:07.003983Z"}}
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# %% [markdown]
# We construct a decision tree model and set up cross-validated grid-search to exhaustively search for the best model parameters
# 
# We will also fit the model to the training set to the model

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:07.006021Z","iopub.execute_input":"2026-02-14T15:11:07.006856Z","iopub.status.idle":"2026-02-14T15:11:11.446520Z","shell.execute_reply.started":"2026-02-14T15:11:07.006820Z","shell.execute_reply":"2026-02-14T15:11:11.445835Z"}}
# Corrected code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Fix: Use a list instead of a set for scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# Now this should work
tree1.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.447549Z","iopub.execute_input":"2026-02-14T15:11:11.447941Z","iopub.status.idle":"2026-02-14T15:11:11.453570Z","shell.execute_reply.started":"2026-02-14T15:11:11.447917Z","shell.execute_reply":"2026-02-14T15:11:11.452674Z"}}
# Check best parameters
tree1.best_params_

# %% [markdown]
# We identify the best AUC score achieved by the decision tree model on the training set

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.454876Z","iopub.execute_input":"2026-02-14T15:11:11.455674Z","iopub.status.idle":"2026-02-14T15:11:11.470964Z","shell.execute_reply.started":"2026-02-14T15:11:11.455638Z","shell.execute_reply":"2026-02-14T15:11:11.469918Z"}}
# Check best AUC score on CV
tree1.best_score_

# %% [markdown]
# This is a very strong AUC score, which shows that this model can predict employees who will leave very well.
# 
# Next, we will write a function that will help us extract all the scores from the grid search.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.472032Z","iopub.execute_input":"2026-02-14T15:11:11.472419Z","iopub.status.idle":"2026-02-14T15:11:11.485092Z","shell.execute_reply.started":"2026-02-14T15:11:11.472394Z","shell.execute_reply":"2026-02-14T15:11:11.484060Z"}}
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table


# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.486289Z","iopub.execute_input":"2026-02-14T15:11:11.486618Z","iopub.status.idle":"2026-02-14T15:11:11.511471Z","shell.execute_reply.started":"2026-02-14T15:11:11.486585Z","shell.execute_reply":"2026-02-14T15:11:11.510622Z"}}
# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results

# %% [markdown]
# All of these scores from the decision tree model are strong indicators of good model performance.
# 
# Recall that decision trees can be vulnerable to overfitting, and random forests avoid overfitting by incorporating multiple trees to make predictions. We will proceed to construct a random forest model next.

# %% [markdown]
# RANDOM FOREST ROUND 1
# 
# Construct a random forest model and set up a cross-validated grid-search to exhuastively search for the best model parameters.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.512628Z","iopub.execute_input":"2026-02-14T15:11:11.513457Z","iopub.status.idle":"2026-02-14T15:11:11.524301Z","shell.execute_reply.started":"2026-02-14T15:11:11.513424Z","shell.execute_reply":"2026-02-14T15:11:11.523174Z"}}
# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {
    "max_depth": [5, None],
    "max_samples": [0.7, 1.0],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 4],
    "n_estimators": [300],
    "max_features": ["sqrt"]  # usually better than 1.0
}

# Assign a dictionary of scoring metrics to capture
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# %% [markdown]
# We fit the model to the training set

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:11:11.525416Z","iopub.execute_input":"2026-02-14T15:11:11.525900Z","iopub.status.idle":"2026-02-14T15:12:42.392375Z","shell.execute_reply.started":"2026-02-14T15:11:11.525876Z","shell.execute_reply":"2026-02-14T15:12:42.391431Z"}}
%%time
rf1.fit(X_train, y_train) # --> Wall time: ~10min

# %% [markdown]
# We are going to save our model 
# 
# We will specify the path to where we want to save our model

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.393680Z","iopub.execute_input":"2026-02-14T15:12:42.394072Z","iopub.status.idle":"2026-02-14T15:12:42.397992Z","shell.execute_reply.started":"2026-02-14T15:12:42.394048Z","shell.execute_reply":"2026-02-14T15:12:42.397191Z"}}
# Define a path to the folder where you want to save the model
path = '/kaggle/working/Salifort_Motors'

# %% [markdown]
# We are going to save our model
# 
# We use Pickle; we are going to define functions to pickle the model and read in the model.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.399203Z","iopub.execute_input":"2026-02-14T15:12:42.399659Z","iopub.status.idle":"2026-02-14T15:12:42.413322Z","shell.execute_reply.started":"2026-02-14T15:12:42.399634Z","shell.execute_reply":"2026-02-14T15:12:42.412273Z"}}
def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.414524Z","iopub.execute_input":"2026-02-14T15:12:42.414953Z","iopub.status.idle":"2026-02-14T15:12:42.427770Z","shell.execute_reply.started":"2026-02-14T15:12:42.414920Z","shell.execute_reply":"2026-02-14T15:12:42.427004Z"}}
def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.428711Z","iopub.execute_input":"2026-02-14T15:12:42.429079Z","iopub.status.idle":"2026-02-14T15:12:42.474951Z","shell.execute_reply.started":"2026-02-14T15:12:42.429046Z","shell.execute_reply":"2026-02-14T15:12:42.473976Z"}}
# Write pickle
write_pickle(path, rf1, 'hr_rf1')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.476111Z","iopub.execute_input":"2026-02-14T15:12:42.476984Z","iopub.status.idle":"2026-02-14T15:12:42.511316Z","shell.execute_reply.started":"2026-02-14T15:12:42.476953Z","shell.execute_reply":"2026-02-14T15:12:42.510593Z"}}
# Read pickle
rf1 = read_pickle(path, 'hr_rf1')

# %% [markdown]
# We identify the best AUC score achieved by the random forest model on the training set.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.512471Z","iopub.execute_input":"2026-02-14T15:12:42.513463Z","iopub.status.idle":"2026-02-14T15:12:42.519124Z","shell.execute_reply.started":"2026-02-14T15:12:42.513436Z","shell.execute_reply":"2026-02-14T15:12:42.518150Z"}}
# Check best AUC score on CV
rf1.best_score_

# %% [markdown]
# We identify the optimal values for the parameters of the random forest model.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.520309Z","iopub.execute_input":"2026-02-14T15:12:42.520632Z","iopub.status.idle":"2026-02-14T15:12:42.536837Z","shell.execute_reply.started":"2026-02-14T15:12:42.520607Z","shell.execute_reply":"2026-02-14T15:12:42.535970Z"}}
# Check best params
rf1.best_params_

# %% [markdown]
# We collect the evaluation scores on the training set for the decision tree and the random forest models

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.537944Z","iopub.execute_input":"2026-02-14T15:12:42.538311Z","iopub.status.idle":"2026-02-14T15:12:42.560532Z","shell.execute_reply.started":"2026-02-14T15:12:42.538287Z","shell.execute_reply":"2026-02-14T15:12:42.559775Z"}}
# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)

# %% [markdown]
# The evaluation scores of the random forest model are better than those of the decision tree model, with the exception of recall (the recall score of the random forest model is approximately 0.001 lower, which is a negligible amount). This indicates that the random forest model mostly outperforms the decision tree model.
# 
# Next, you can evaluate the final model on the test set.

# %% [markdown]
# Define a function that gets all the scores from a model's predictions.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.561625Z","iopub.execute_input":"2026-02-14T15:12:42.562012Z","iopub.status.idle":"2026-02-14T15:12:42.577040Z","shell.execute_reply.started":"2026-02-14T15:12:42.561987Z","shell.execute_reply":"2026-02-14T15:12:42.576194Z"}}
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table

# %% [markdown]
# We will use the best-performing model to predict on the test set

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.577836Z","iopub.execute_input":"2026-02-14T15:12:42.578098Z","iopub.status.idle":"2026-02-14T15:12:42.722973Z","shell.execute_reply.started":"2026-02-14T15:12:42.578076Z","shell.execute_reply":"2026-02-14T15:12:42.722095Z"}}
# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores

# %% [markdown]
# The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, we are more confident that our model's performance on this data is representative of how it will perform on new, unseen data.

# %% [markdown]
# FEATURE ENGINEERING
# 
# We might be skeptical of the high evaluation scores. There is a chance that there is some data leakage occurring. Data leakage is when you use data to train your model that should not be used during training, either because it appears in the test data or because it's not data that you'd expect to have when the model is actually deployed. Training a model with leaked data can give an unrealistic score that is not replicated in production.
# In this case, it's likely that the company won't have satisfaction levels reported for all of its employees. It's also possible that the `average_monthly_hours` column is a source of some data leakage. If employees have already decided to quit or have already been identified by management as people to be fired, they may be working fewer hours.
# The first round of decision tree and random forest models included all variables as features. This next round will incorporate feature engineering to build improved models.
# We could proceed by dropping `satisfaction_level` and creating a new feature that roughly captures whether an employee is overworked. We could call this new feature `overworked`. It will be a binary variable.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.724084Z","iopub.execute_input":"2026-02-14T15:12:42.724423Z","iopub.status.idle":"2026-02-14T15:12:42.744095Z","shell.execute_reply.started":"2026-02-14T15:12:42.724390Z","shell.execute_reply":"2026-02-14T15:12:42.743316Z"}}
# Drop `satisfaction_level` and save resulting dataframe in new variable
df1 = df_enc.drop('satisfaction_level', axis=1)

# Display first few rows of new dataframe
df1.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.745191Z","iopub.execute_input":"2026-02-14T15:12:42.745541Z","iopub.status.idle":"2026-02-14T15:12:42.753327Z","shell.execute_reply.started":"2026-02-14T15:12:42.745516Z","shell.execute_reply":"2026-02-14T15:12:42.752173Z"}}
# Create `overworked` column. For now, it's identical to average monthly hours.
df1['overworked'] = df1['average_monthly_hours']

# Inspect max and min average monthly hours values
print('Max hours:', df1['overworked'].max())
print('Min hours:', df1['overworked'].min())

# %% [markdown]
# 166.67 is approximately the average number of monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day.
# 
# We define being overworked as working more than 175 hours per month on average.
# 
# To make the overworked column binary, we could reassign the column using a boolean mask.

# %% [markdown]
# df3['overworked'] > 175 creates a series of booleans, consisting of True for every value > 175 and False for every value ≤ 175
# .astype(int) converts all True to 1 and all False to 0

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.754577Z","iopub.execute_input":"2026-02-14T15:12:42.754881Z","iopub.status.idle":"2026-02-14T15:12:42.770509Z","shell.execute_reply.started":"2026-02-14T15:12:42.754854Z","shell.execute_reply":"2026-02-14T15:12:42.769204Z"}}
# Define `overworked` as working > 175 hrs/week
df1['overworked'] = (df1['overworked'] > 175).astype(int)

# Display first few rows of new column
df1['overworked'].head()

# %% [markdown]
# We drop the average monthly hours column

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.771705Z","iopub.execute_input":"2026-02-14T15:12:42.772169Z","iopub.status.idle":"2026-02-14T15:12:42.796802Z","shell.execute_reply.started":"2026-02-14T15:12:42.772125Z","shell.execute_reply":"2026-02-14T15:12:42.796047Z"}}
# Drop the `average_monthly_hours` column
df1 = df1.drop('average_monthly_hours', axis=1)

# Display first few rows of resulting dataframe
df1.head()

# %% [markdown]
# We again isolate the features and target variables

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.797949Z","iopub.execute_input":"2026-02-14T15:12:42.798304Z","iopub.status.idle":"2026-02-14T15:12:42.804998Z","shell.execute_reply.started":"2026-02-14T15:12:42.798270Z","shell.execute_reply":"2026-02-14T15:12:42.804072Z"}}
# Isolate the outcome variable
y = df1['left']

# Select the features
X = df1.drop('left', axis=1)

# %% [markdown]
# We split the data into training and testing sets

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.806064Z","iopub.execute_input":"2026-02-14T15:12:42.806389Z","iopub.status.idle":"2026-02-14T15:12:42.833091Z","shell.execute_reply.started":"2026-02-14T15:12:42.806356Z","shell.execute_reply":"2026-02-14T15:12:42.832108Z"}}
# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# %% [markdown]
# DECISION TREE ROUND 2 

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:42.834398Z","iopub.execute_input":"2026-02-14T15:12:42.835038Z","iopub.status.idle":"2026-02-14T15:12:46.764220Z","shell.execute_reply.started":"2026-02-14T15:12:42.835014Z","shell.execute_reply":"2026-02-14T15:12:46.763477Z"}}
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 4, 6]
}
# Assign a dictionary of scoring metrics to capture
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}
# Instantiate GridSearch
tree2 = GridSearchCV(
    tree,
    cv_params,
    scoring=scoring,
    cv=4,
    refit="roc_auc",
    n_jobs=-1,
    verbose=1
)
tree2.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:46.765246Z","iopub.execute_input":"2026-02-14T15:12:46.765606Z","iopub.status.idle":"2026-02-14T15:12:46.770985Z","shell.execute_reply.started":"2026-02-14T15:12:46.765582Z","shell.execute_reply":"2026-02-14T15:12:46.770095Z"}}
# Check best params
tree2.best_params_

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:46.771898Z","iopub.execute_input":"2026-02-14T15:12:46.772259Z","iopub.status.idle":"2026-02-14T15:12:46.788046Z","shell.execute_reply.started":"2026-02-14T15:12:46.772236Z","shell.execute_reply":"2026-02-14T15:12:46.787222Z"}}
# Check best AUC score on CV
tree2.best_score_

# %% [markdown]
# This model performs very well, even without satisfaction levels and detailed hours worked data.
# 
# Next, check the other scores.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:46.789009Z","iopub.execute_input":"2026-02-14T15:12:46.789220Z","iopub.status.idle":"2026-02-14T15:12:46.809448Z","shell.execute_reply.started":"2026-02-14T15:12:46.789201Z","shell.execute_reply":"2026-02-14T15:12:46.808462Z"}}
# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)

# %% [markdown]
# Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.

# %% [markdown]
# RANDOM FOREST - ROUND 2

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:12:46.810390Z","iopub.execute_input":"2026-02-14T15:12:46.810676Z","iopub.status.idle":"2026-02-14T15:16:51.604325Z","shell.execute_reply.started":"2026-02-14T15:12:46.810649Z","shell.execute_reply":"2026-02-14T15:16:51.603551Z"}}
# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {
    "max_depth": [3, 5, None],
    "max_features": ["sqrt"],        # usually better than 1.0 for RF
    "max_samples": [0.7, 1.0],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [2, 3, 4],
    "n_estimators": [300, 500],
}

# Assign a dictionary of scoring metrics to capture (MUST be dict, not set)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Instantiate GridSearch
rf2 = GridSearchCV(
    rf,
    cv_params,
    scoring=scoring,
    cv=4,
    refit="roc_auc",
    n_jobs=-1,     # speeds up training
    verbose=1
)
# Fit the model
rf2.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:16:51.606059Z","iopub.execute_input":"2026-02-14T15:16:51.606403Z","iopub.status.idle":"2026-02-14T15:16:51.639953Z","shell.execute_reply.started":"2026-02-14T15:16:51.606370Z","shell.execute_reply":"2026-02-14T15:16:51.639180Z"}}
# Write pickle
write_pickle(path, rf2, 'hr_rf2')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:16:51.640996Z","iopub.execute_input":"2026-02-14T15:16:51.641389Z","iopub.status.idle":"2026-02-14T15:16:51.676642Z","shell.execute_reply.started":"2026-02-14T15:16:51.641353Z","shell.execute_reply":"2026-02-14T15:16:51.675696Z"}}
# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:16:51.677624Z","iopub.execute_input":"2026-02-14T15:16:51.677920Z","iopub.status.idle":"2026-02-14T15:16:51.683863Z","shell.execute_reply.started":"2026-02-14T15:16:51.677896Z","shell.execute_reply":"2026-02-14T15:16:51.683076Z"}}
# Check best params
rf2.best_params_

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:16:51.685305Z","iopub.execute_input":"2026-02-14T15:16:51.686154Z","iopub.status.idle":"2026-02-14T15:16:51.704021Z","shell.execute_reply.started":"2026-02-14T15:16:51.686128Z","shell.execute_reply":"2026-02-14T15:16:51.703119Z"}}
# Check best AUC score on CV
rf2.best_score_

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:16:51.705211Z","iopub.execute_input":"2026-02-14T15:16:51.705600Z","iopub.status.idle":"2026-02-14T15:16:51.726613Z","shell.execute_reply.started":"2026-02-14T15:16:51.705577Z","shell.execute_reply":"2026-02-14T15:16:51.725707Z"}}
# Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)

# %% [markdown]
# Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric.
# 
# We proceed to score the champion model on the test set now.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:18:32.834479Z","iopub.execute_input":"2026-02-14T15:18:32.835011Z","iopub.status.idle":"2026-02-14T15:18:32.980275Z","shell.execute_reply.started":"2026-02-14T15:18:32.834975Z","shell.execute_reply":"2026-02-14T15:18:32.979204Z"}}
# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
rf2_test_scores

# %% [markdown]
# This seems to be a stable, well-performing final model.
# 
# Plot a confusion matrix to visualize how well it predicts on the test set.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:19:01.594429Z","iopub.execute_input":"2026-02-14T15:19:01.594926Z","iopub.status.idle":"2026-02-14T15:19:01.872062Z","shell.execute_reply.started":"2026-02-14T15:19:01.594896Z","shell.execute_reply":"2026-02-14T15:19:01.870682Z"}}
# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)
disp.plot(values_format='');

# %% [markdown]
# DECISION TREE SPLITS

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:20:27.426453Z","iopub.execute_input":"2026-02-14T15:20:27.427539Z","iopub.status.idle":"2026-02-14T15:20:30.056916Z","shell.execute_reply.started":"2026-02-14T15:20:27.427506Z","shell.execute_reply":"2026-02-14T15:20:30.055470Z"}}
# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()

# %% [markdown]
# DECISION TREE FEATURE IMPORTANCE

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:21:32.409865Z","iopub.execute_input":"2026-02-14T15:21:32.410233Z","iopub.status.idle":"2026-02-14T15:21:32.421648Z","shell.execute_reply.started":"2026-02-14T15:21:32.410205Z","shell.execute_reply":"2026-02-14T15:21:32.420924Z"}}
#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances

# %% [markdown]
# We then create a barplot to visualize the decision tree feature importances.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:22:48.420016Z","iopub.execute_input":"2026-02-14T15:22:48.420828Z","iopub.status.idle":"2026-02-14T15:22:48.621937Z","shell.execute_reply.started":"2026-02-14T15:22:48.420791Z","shell.execute_reply":"2026-02-14T15:22:48.621116Z"}}
sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()

# %% [markdown]
# The barplot above shows that in this decision tree model, last_evaluation, number_project, tenure, and overworked have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left.

# %% [markdown]
# RANDOM FOREST FEATURE IMPORTANCE
# 
# We plot the feature importance for the random forest model

# %% [code] {"execution":{"iopub.status.busy":"2026-02-14T15:25:21.456860Z","iopub.execute_input":"2026-02-14T15:25:21.457671Z","iopub.status.idle":"2026-02-14T15:25:21.686370Z","shell.execute_reply.started":"2026-02-14T15:25:21.457638Z","shell.execute_reply":"2026-02-14T15:25:21.685408Z"}}
# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features 
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()

# %% [markdown]
# The plot above shows that in this random forest model, last_evaluation, number_project, tenure, and overworked have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left, and they are the same as the ones used by the decision tree model.

# %% [markdown]
# ## CONCLUSIONS AND RECOMMENDATION
# The models and the feature importances extracted from the models confirm that employees at the company are overworked.
# To retain employees, the following recommendations could be presented to the stakeholders:
# 
# Cap the number of projects employees can work on.
# 
# Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
# 
# If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
# 
# Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
# 
# High evaluation scores should not be reserved for employees who work 200+ hours per month.
# 
# Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
# 

# %% [markdown]
# ## NEXT STEPS
# It may be justified to still have some concern about data leakage. It could be prudent to consider how predictions change when `last_evaluation` is removed from the data. It's possible that evaluations aren't performed very frequently, in which case it would be useful to be able to predict employee retention without this feature. It's also possible that the evaluation score determines whether an employee leaves or stays, in which case it could be useful to pivot and try to predict performance score. The same could be said for satisfaction score.
# 
# For another project, consider building a K-means model on this data and analyzing the clusters. This approach may reveal valuable insights.
