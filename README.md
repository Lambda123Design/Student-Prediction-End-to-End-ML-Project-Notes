# Student-Prediction-End-to-End-ML-Project-Notes

**A) Github And Code Setup**

**B) Project Structure, Logging and Exception**

**C) Project Problem Statement EDA And Model Training**

**D) Data Ingestion Implementation**

**E) Data Transformation Implementation**

# **A) Github And Code Setup**

So guys, we are about to start the end to end machine learning implementation, web deployment. And again the main aim of this specific series is to develop end to end projects and the things that we are going to learn. The similar things are basically applied in the industry when we are solving any kind of projects. Uh, we are going to basically discuss in this series about each and every module, what all things we can basically do on top of it, and how we can easily crack any data science interviews. This end to end project implementation will be a game changer for cracking data science interviews. Why? Because all the students who have till now cracked, or any professionals who have cracked data science interviews, they have really explained the projects very well.

Okay, so from your side, only one thing is basically required. That is dedication. If you are able to dedicate properly, if you are able to see, if you are able to understand, if you are able to practice right, how the project implementation is done, and then you can probably apply in different different projects like in deep learning and all, all the techniques will be same. We'll try to create a generic project structure. We'll try to use methodologies, tools, techniques in such a way that you will be able to apply in any kind of projects going forward.

Now what all things we are going to do, what all things we are planning to do, everything we will be discussing. How I'm going to divide this particular parts of all this particular series, that also I'll be discussing. So let me go ahead and let me share my screen. So, uh, all the line by line code I will be writing it in front of you will be implemented in such a way that parallelly we will try to implement, I will be committing the code, and then based on that, we will go ahead with it.

Okay. So let's go ahead and let's try to do this. Now I will just open this one. So obviously I've spoken about the agenda. The agenda of today's video is that whenever you are doing any kind of machine learning or deep learning project, the first and the most important thing is that it's all about your repository, right? where you are committing the code because you specifically work in a team, right? A team of people.

Now, when you are working with a team of people, definitely you will be seeing that there are many people who will be collaborating right at the same time. They will be committing the code. They will be merging the code. They will be developing new models. So the first thing is that what we are going to do is that we are going to basically set up the GitHub. Right. And this GitHub will be our repository, right, for committing our code. And probably if you are working in a team, you can also work in a team so that your development will be in sync whenever you are probably developing it. Right. The entire application in short. So setting up the GitHub will be the first one.

Then, uh, in this GitHub, what are the best things that we will be applying? Like let's say in the first case I want to go ahead and probably set up a new environment. Right. So GitHub and this is today's agenda. What I'm going to basically say new environment, right. How do we create a new environment. What all things you have to make sure that, uh, while working with a new environment, there are some things that we really need to make sure that that thing is in place before implementing the project. It is always a good practice.

Then, uh, we will try to decide a mini project structure. Mini project structure. Mini project structure. I will not say mini anyhow, I'm actually developing the entire project, right? We'll be discussing also about setup, right. setup.py. Now what is the importance of setup.py? We'll try to understand after creating the new environment. And finally, how probably I can also build a package using setup.py. We'll also discuss about requirements.txt. So, uh, again there is a relationship between requirements.txt and setup.py. Also we will discuss about that.

We'll create a new environment and we'll do this. We'll do all these tasks today. And in the next video, what we are basically going to do is that we are going to go with the next step wherein we will be implementing logging, exception handling. And then we will try to decide a formal project structure. How do we decide a proper, uh, project structure? That also I'll be talking about what are the best practices that we specifically follow.

Now let's go with the first one, that is, set up the GitHub repository. So what I will do, I will quickly go into my GitHub. Uh, all the committing of the code will happen in the GitHub, guys. This will basically give us a real-world industry experience. In short, because we are continuously working, we are committing the code, we are merging the code, we are pulling the code, and many things we are going to do with respect to that.

So the first thing first, I will go to my repositories over here. So I hope everybody has a GitHub account. So let me go ahead and create a new repository. Now with respect to this new repository, I'm just going to give some name. So let's say my name will be "ML_projects". Okay. As I'm saying that this will be a generic completely generic project structure. Later on you can replace any dataset. We will try to use all those techniques. What all techniques is basically required that will follow. So "ML_projects".

So right now I'll keep it as public so that I will be able to share this with you. Okay. And then let's go ahead. Don't click anything as such right now, okay. Don't think, don't click anything. Just write "ML_projects". No, don't even write description. I'll show you how we can basically write everything and all. So I will just go ahead and click "Create repository".

Now once you create the repository, this is what you specifically get, okay. Now this is about my ML project. Now, what I did is that I will probably go into my E drive, okay. So I have created a folder. Now inside this I will be doing my entire project development. Okay. So I will copy this path. I'll open Anaconda prompt, okay. Now the reason why I'm opening Anaconda prompt, I will just go to this specific path. So let's go to E drive. So I'm in this particular path that is "ML_projects", right now.

Here you can see that I'm in the base environment. Right. And obviously, uh, if you have Anaconda prompt or if you have VSCode, anything as such. Okay. We can actually start again to talk about. Many people are also talking about the prerequisites. You really need to know Python programming language, model or coding. You need to know about machine learning algorithm. So if you are dedicatedly following my machine learning playlist or the deep learning playlist or the Python programming language, it will be more than sufficient.

Now from here, what I'm actually going to do is that I'm just going to type "code .". Now this "code ." will basically launch a VSCode instance. Okay. So if I execute this, this is how my VSCode instance will look like. Okay. So here you can probably see this. It will show "GitHub could not be connected". It's okay. We can just cut this. So this is an empty project completely initially to start with. And this is my VSCode that you will be able to see. Okay, this is my GitHub repository. Okay. And this is my VSCode. Right now, first thing first, I have to make sure that I sync with my GitHub repository over here.

Okay. So to start with, what I'm actually going to do over here is that I will just go ahead and open a terminal. So here is my new terminal. Okay. So this is my new terminal, right? It can be a PowerShell. Or I can also use Command Prompt. Right. It is up to us, right? Whatever terminal that you want to specifically use, okay. Now the first step is always to create an environment, right?

Now, what I want is that guys, I don't want to create a separate environment, but instead what I'll do here, only an environment will be getting created. And whatever packages I will be installing, that will come inside this particular folder. Okay, so let me just show you what I will do. So basically what I'm saying is that inside this particular project itself, my environment will get created, and whatever packages I install, that all will get created over here. Okay. So this is where, uh, I'm going to make sure that all my packages are available over here. And this is a good practice because at the end of the day, I can freeze all my libraries from this particular folder itself.

Now, after doing this, I'm just going to clear my screen. So I'll write "CLS" and what I'm actually going to do, I'll just write "conda activate venv". Okay. "Venv" is my environment. And I'll just write like this. So automatically you'll be able to see that I will be inside this particular environment. So I have activated my environment in short. Okay, so perfect. We have done this. We have actually created our environment. We have done everything as such. Okay. This is perfect till here.

Now the next thing what we are going to do is that we are going to clone this entire repository and we need to sync this with the GitHub so that we will be able to commit all our code. Okay. So we will just follow this step by step and we will see how we can make sure that whatever things we are committing, we basically commit in this specific repository. So first of all, we really need to initialize git. Okay. So I will go over here, open this project, so let me go ahead and write "git init". Okay. So once I write this, I will be initializing an empty git repository in this specific location. Okay. So you'll be able to find out this particular folder. Right now it is hidden. Okay. But if you go into the folders itself, you'll be able to see it. Perfect.

This is the first portion converted into multiple paragraphs with Python code included in quotes.

# Notes:

1. Introduction to End-to-End ML Project Implementation

In this series, we are going to implement an end-to-end machine learning project including web deployment. The main goal is to develop practical projects that mirror real-world industry scenarios. The techniques and methodologies used here are the same ones applied when solving projects in professional settings. By following this series, you will learn how to implement projects step-by-step, which will be extremely useful for cracking data science interviews.

The success in interviews often depends on how well you can explain your projects, and this series will give you that confidence. The only requirement from your side is dedication — paying attention, understanding concepts, and practicing consistently. Once you grasp the project workflow, the same principles can be applied to other areas, including deep learning projects.

2. Learning Objectives and Methodology

In this series, we will discuss each module in detail, explore best practices, tools, and techniques, and create a generic project structure. This will allow you to adapt the workflow to any type of machine learning or deep learning project. The goal is to develop a strong understanding of project setup, coding workflow, and real-world practices that can be reused in different projects.

All code will be written line by line, committed to a GitHub repository, and implemented simultaneously to provide a hands-on learning experience.

3. Importance of GitHub Repository

The first and most important aspect of any project is version control. When working in a team, multiple people collaborate, committing, merging, and developing new models. To manage this, we start by setting up a GitHub repository. This repository will be used to store all project code, track changes, and ensure that team members remain synchronized.

4. Creating the GitHub Repository

To begin, go to your GitHub account and create a new repository. For example, the repository name could be "ML_projects" for a generic project structure. Initially, you can set it as public. We will not add a description yet; this will be shown later. Once created, GitHub will provide a repository link and a structure ready to be cloned.

5. Setting Up Local Project Folder

On your local machine, create a folder (e.g., "ML_projects") in your preferred drive. Then open Anaconda Prompt and navigate to this folder:

E:
cd ML_projects

Once in the folder, you can launch VSCode for the project:

code .

This will open VSCode in the current folder. Initially, it will be an empty project, and GitHub may show a message like "GitHub could not be connected" — this is okay.

6. Creating a Virtual Environment

The next step is to create a virtual environment for the project. This ensures that all dependencies are contained within the project folder, making it portable and easy to manage.

conda create -n venv python=3.10
conda activate venv

Now, the environment "venv" is active. All packages installed from this point will be confined to this environment. You can also freeze the libraries later using requirements.txt.

7. Initializing Git in the Project

To sync the local folder with GitHub, first initialize Git in the project folder:

git init

This will create a hidden .git folder that tracks all changes in your project. Once initialized, you can commit and push code to GitHub. This step gives you a real-world industry experience of continuous development, version control, and collaboration.

# **B) Project Structure, Logging and Exception**

So we are going to continue the discussion with respect to the end-to-end ML project implementation.

In our previous session, we have already done many things over here. We have set up our setup.py file. You understood the purpose of this, right? To basically create the package. We have this requirements.txt file. So any packages that will be required, I’ll probably be writing it over here. We had seen how to probably create an environment and how to start everything. We have actually done it till now.

Now in this video, we are going to focus on creating the entire project structure and then some of the common functionalities like logging, exception handling, how to write that specific code, where you should specifically write it, what should be your project structure, everything that we’ll be doing in this specific session.

Okay, so in the source folder right now, I just have __init__.py.

Now guys, see, whenever right now when I’m creating the project structure, I will just do it manually because I need to show everything for you from scratch later on. This entire process can also be automated. I just need to write one code like template.py, and automatically the entire folder structure will be created, right? I just need to write some 10 to 15 lines of code. But right now I am showing you everything from scratch so that you understand the project well.

So in the source, because whatever project I’m implementing, everything will come inside this. In the source, we are going to create some very important folders.

The first folder that we are going to create is something called components. Right now, what exactly these components are and what will basically come over here, we’ll discuss that. But before, as soon as we create a folder, the first thing that you really need to do is create the file __init__.py.

Why do we do this? Because components will be created as a package and it can also be exported; it can be imported to some other file location. That is the reason I’m writing this __init__.py.

Okay. Now these components are like all the modules that we are probably going to create. Like initially we will probably create a process called data ingestion. Data ingestion basically means I will be reading a dataset from a database, or it can be from some other file locations, or it can be from different kinds of databases also.

So initially we need to read the database. Reading the data from a specific database is specifically called data ingestion. So data ingestion is a part of a module when we are developing the entire project. That can be our component. So let’s say I’m going to create a component over here, and the first component that we are probably going to create is called data_ingestion.

For this, I will just create my .py file. This will basically have all the code that is something related to reading the data. Here I have my data ingestion. After reading the data, what we will be doing, we may probably do data validation. We may probably do data transformation.

Since I really want to make this project very precise and very easy for you to understand, after ingesting the data, I will try to transform that particular data. So my next step will be something called data_transformation. This will also be a .py file. Any code that is related to transformation, like how to change categorical features into numerical features, how to handle one-hot encoding, label encoding, and other preprocessing steps, will be written over here.

So these are the two steps: first, data ingestion, and second, data transformation. During data ingestion, we will also divide the dataset into train and test for training the model. Then we may also create a validation dataset. All the code for that will be in data_ingestion.py.

After doing data transformation, the next step is model_trainer. Here we will specifically train the model. All the training code, different model selections, evaluation metrics like confusion matrix for classification or R², adjusted R² for regression, all those things will be handled here. From here, we could also implement a model_pusher for pushing the trained model to cloud storage, but for this first project, we’ll keep it simple.

These components—data_ingestion, data_transformation, model_trainer—are mainly for the training purpose. This folder structure is specifically called components. Components are the modules that we will specifically use in the project.

Now, I’ll create one more folder called pipeline. We will create two types of pipelines: training pipeline and prediction pipeline. Inside pipeline, I’ll create train_pipeline.py, which will have all the code for the training pipeline and will call the components. Next, I’ll create predict_pipeline.py for prediction purposes. We’ll also create __init__.py inside the pipeline folder for imports.

Next, since the entire project implementation happens inside source, I will create three important files: one for logging—logger.py, one for exception handling—exception.py, and one for utility functions—utils.py. Utilities can include functions like reading a dataset from a database, creating a MongoDB client, saving a model to cloud, etc., which will be used across components.

If you understand this project structure, no one can stop you from understanding the entire project. That is why I am going step by step.

Now let’s write the exception handling code. For exceptions, I will write a custom exception. First, I’ll import sys. The sys module provides functions and variables to manipulate different parts of the Python runtime environment. Any exception that occurs can be controlled and tracked using sys.

I’ll define a function called error_message_details(error, error_detail) with two parameters: the error message and error details from sys. error_detail.exc_info() provides execution information, including the traceback. From the traceback, we can get the file name, line number, and error message. Using this, we create a formatted error string.

Next, I’ll create a class CustomException that inherits from Python’s Exception. In the constructor, I initialize error_message using the function above and store error_detail for tracking. I also override the __str__ method to return the error message whenever the exception is printed. This custom exception class can now be used throughout the project wherever try-except blocks are needed.

Now for logging, we import logging, os, and datetime. We create a log file path using the current working directory and a timestamped file name. We create directories if they do not exist and set up logging.basicConfig to specify the log file name, format, and logging level. The format includes timestamp, line number, module name, level, and message. Any logs written using logging.info will follow this format and be stored in the log file.

Finally, the utils.py file will hold reusable functions for the project, like reading datasets, saving models, connecting to databases, etc.

At this stage, our project structure is complete with components, pipelines, logger, exception handling, and utilities. This ensures modular programming, clean code, and industry-standard practices.

To test logging, you can add a simple logging.info("Logging has started") in logger.py and run it to check if the log file is created. For exception handling, use a try-except block with a deliberate error like division by zero and raise the CustomException. You should see the custom exception message with file name, line number, and error description.

Once verified, you can commit the files to Git using git add ., git commit -m "logging and exception", and git push -u origin main. Now your repository contains the project structure, exception handling, logger setup, and utility file, ready for the next steps in the ML project.

In the next video, we will discuss the problem statement, take the dataset, perform EDA, and start coding data ingestion, transformation, and model training. We will also demonstrate how to read data from a database like MongoDB to give you an idea of real-world data ingestion.

This concludes the project structure, logging, and exception setup tutorial. Make sure to implement it yourself and share your GitHub code on LinkedIn for feedback.

# Notes:

1. Recap of Previous Session

In the previous session, we have already covered several foundational tasks: setting up the setup.py file to create a Python package, writing a requirements.txt file for dependency management, and creating a virtual environment. We have completed all the initial setup, which forms the foundation for building the full ML project.

2. Focus of This Session

In this session, we will focus on creating the entire project structure and implementing some common functionalities such as logging and exception handling. We will also learn where to write code, how to organize files, and best practices for a modular and scalable project.

3. Creating the Source Folder

Currently, the source folder contains only __init__.py. This is necessary to make the folder a Python package so it can be imported elsewhere. We will manually create the folder structure from scratch, although this process can be automated with a script (template.py).

4. Creating the Components Folder

Inside source, we create a folder called components. This folder will contain all modules used for the ML project. Each module will have its own .py file and an __init__.py for package exports.

Components include:

data_ingestion.py – Code for reading datasets from databases, file locations, or other sources. This module may also split the dataset into train, test, and validation sets.

data_transformation.py – Code for preprocessing and transforming data, such as handling categorical features, one-hot encoding, label encoding, or scaling numerical features.

model_trainer.py – Code for training models, selecting algorithms, and evaluating metrics like confusion matrix (classification) or R² (regression).

Optionally, a model_pusher module could be added for pushing trained models to cloud storage, but for the first project, this will remain simple.

5. Creating the Pipeline Folder

We create a pipeline folder inside source. This folder will include:

train_pipeline.py – Handles the full training workflow and calls relevant components.

predict_pipeline.py – Handles prediction workflows.

__init__.py – To make the folder a Python package.

The pipeline folder links all components in a structured and reusable way.

6. Creating Core Utility Files

Inside source, we create three additional core files:

logger.py – For logging messages across the project.

exception.py – For custom exception handling.

utils.py – For utility functions like reading datasets, saving models, connecting to databases, etc.

This setup ensures modular programming, clean code, and industry-standard practices.

7. Exception Handling Setup

In exception.py, we create a custom exception class:

import sys

def error_message_details(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name} at line {line_number}: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail=sys):
        self.error_message = error_message_details(error, error_detail)
        self.error_detail = error_detail
    def __str__(self):
        return self.error_message

This class captures the file name, line number, and error message whenever an exception occurs. It can be used throughout the project with try-except blocks.

8. Logging Setup

In logger.py, we set up logging using the logging module:

import logging
import os
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Logging has started")

This setup ensures all logs are stored in a timestamped file, with details like timestamp, log level, and message.

9. Utility Functions Setup

utils.py will hold reusable functions such as:

Reading datasets from databases

Saving models to cloud storage

Creating database clients (MongoDB, PostgreSQL, etc.)

This ensures that common functionality is centralized and reusable across the project.

10. Testing Logging and Exception Handling

To verify:

Logging – Add logging.info("Test log message") and run logger.py. Check the log file is created.

Exceptions – Test using a deliberate error:

try:
    1/0
except Exception as e:
    raise CustomException(e, sys)


You should see a formatted error message with file name, line number, and description.

11. Committing to GitHub

Once verified, commit the project structure and utility setup to GitHub:

git add .
git commit -m "Added project structure, logging, and exception handling"
git push -u origin main

The repository now contains the full project structure, exception handling, logger setup, and utility functions, ready for the next steps: problem statement, dataset, EDA, data ingestion, transformation, and model training.

This ensures your ML project is modular, professional, and aligned with industry practices.

**C) Project Problem Statement EDA And Model Training**

So guys, in the previous tutorial we implemented exception handling with logging for the entire project. The main issue we fixed was that exceptions were not getting saved in the log file because I had not imported the logger. The fix is to import it using "from src.logger.logger import logging" and then you can log an error like "logging.info('division by zero')" and run your script "python source/exception.py". You’ll see the error logged in the log file.

Now the agenda of this session is to start a project on Student Performance Indicator. The dataset contains categorical and numerical features, some NaN values, and multiple types of columns. The dataset "student.csv" has 1000 rows and 8 columns, including features like "gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "math score", "reading score", "writing score". Our goal is to predict student test scores using these features.

For EDA, we first read the data with "import pandas as pd" and "df = pd.read_csv('notebook/student.csv')". We perform checks like:

"df.isnull().sum()" for missing values

"df.duplicated().sum()" for duplicates

"df.info()" for data types

"df.describe()" for statistics

"df['gender'].unique()" for unique values in categorical columns

We can also separate numerical and categorical features with:
"numerical_features = [feature for feature in df.columns if df[feature].dtype != 'object']"
"categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']"

For feature engineering, we create total and average scores:
"df['total_score'] = df['math score'] + df['reading score'] + df['writing score']"
"df['average_score'] = df['total_score'] / 3"

For visualization, you can use histograms or group by plots, e.g.:
"df.groupby('gender')['average_score'].mean().plot(kind='bar')"

Next, for model training, we define X and y:
"X = df.drop(['math score'], axis=1)"
"y = df['math score']"

We preprocess data with column transformer pipelines:

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_features = X.select_dtypes(exclude='object').columns
categorical_features = X.select_dtypes(include='object').columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

We apply the transformations with "X_processed = preprocessor.fit_transform(X)". Train-test split can be done with:
"from sklearn.model_selection import train_test_split"
"X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)"

For model evaluation, define a function to return MAE, MSE, RMSE, R²:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }

Train multiple regression models like Linear Regression, Ridge, RandomForest, CatBoost, XGBoost using:

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'CatBoost': CatBoostRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, evaluate_model(y_test, y_pred))

Based on R² or RMSE, choose the best model and make predictions. Later, we will convert all this into modular coding, moving functions like train-test split, model evaluation, and training into "utils.py" or "model_trainer.py" to map the workflow properly.

Finally, after completing the code, you can commit and push to GitHub using:
"git add ."
"git commit -m 'EDA and problem statement'"
"git push -u origin main"

This completes the initial end-to-end implementation from EDA, feature engineering, preprocessing, modeling, evaluation, and finally pushing code to GitHub.

# Notes:

1. Recap of Previous Session

In the previous tutorial, we implemented exception handling with logging for the entire project. The main issue we fixed was that exceptions were not being saved in the log file because the logger had not been imported. The fix is to import it using "from src.logger.logger import logging". After that, you can log an error like "logging.info('division by zero')" and run your script using "python source/exception.py". You’ll then see the error saved in the log file.

2. Agenda of This Session

In this session, we will start a project on Student Performance Indicator. The dataset "student.csv" contains both categorical and numerical features, some NaN values, and multiple types of columns. It has 1000 rows and 8 columns, including features like "gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "math score", "reading score", and "writing score". Our goal is to predict student test scores using these features.

3. Exploratory Data Analysis (EDA)

For EDA, we first read the data using: "import pandas as pd" and "df = pd.read_csv('notebook/student.csv')". We perform basic checks such as:

"df.isnull().sum()" for missing values

"df.duplicated().sum()" for duplicates

"df.info()" for data types

"df.describe()" for statistical summaries

"df['gender'].unique()" for unique values in categorical columns

We can also separate numerical and categorical features with:
"numerical_features = [feature for feature in df.columns if df[feature].dtype != 'object']"
"categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']"

4. Feature Engineering

For feature engineering, we create total and average scores:
"df['total_score'] = df['math score'] + df['reading score'] + df['writing score']"
"df['average_score'] = df['total_score'] / 3"

For visualization, you can use histograms or group-by plots, e.g.:
"df.groupby('gender')['average_score'].mean().plot(kind='bar')"

5. Preparing Data for Model Training

We define the features and target variable as:
"X = df.drop(['math score'], axis=1)"
"y = df['math score']"

For preprocessing, we use column transformer pipelines:

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_features = X.select_dtypes(exclude='object').columns
categorical_features = X.select_dtypes(include='object').columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

We apply transformations using: "X_processed = preprocessor.fit_transform(X)".

6. Train-Test Split

Split the dataset into training and testing sets using:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

7. Model Evaluation Function

Define a function to evaluate models with metrics MAE, MSE, RMSE, and R²:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }

8. Training Multiple Regression Models

Train multiple models like Linear Regression, Ridge, RandomForest, AdaBoost, CatBoost, and XGBoost:

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'CatBoost': CatBoostRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, evaluate_model(y_test, y_pred))

Based on R² or RMSE, choose the best model for predictions.

9. Modularizing the Code

Later, we will modularize the workflow by moving functions like train-test split, model evaluation, and model training into utils.py or model_trainer.py. This will help map the workflow properly and maintain clean code.

10. Committing to GitHub

Once the code is complete, commit and push to GitHub:

git add .
git commit -m "EDA and problem statement"
git push -u origin main

At this stage, the repository will contain EDA, feature engineering, preprocessing, modeling, evaluation, and GitHub updates.

This completes the initial end-to-end ML implementation from EDA to modeling and GitHub integration.

# **D) Data Ingestion Implementation**

