Mini Project 1
Graded Assessment: Mini Project 1

In this section, you will be given a set of data, which contains statistics for news articles retrieved from an online news website. The attributes are described in the table below.

List of Attributes

Attribute

Description

ID

Unique ID for each news article

URL

URL of the news article

timedelta

Number of days between the article's publication and the dataset acquisition

weekday

The day of the week when the article was published

n_tokens_title

Number of words in the title

n_tokens_content

Number of words in the content

n_unique_tokens

Rate of unique words in the content

n_non_stop_words

Rate of non-stop words in the content

n_non_stop_unique_tokens

Rate of unique non-stop words in the content

num_hrefs

Number of links

num_self_hrefs

Number of links to other articles published by the news website

num_imgs

Number of images

num_videos

Number of videos

n_comments

Number of comments by readers

average_token_length

Average length of the words in the content

data_channel

The data channel of the article

self_reference_min_shares

Minimum shares of referenced articles in the news website

self_reference_max_shares

Maximum shares of referenced articles in the news website

self_reference_avg_shares

Average shares of referenced articles in the news website

num_keywords

Number of keywords in the metadata

kw_min_min

Minimum shares any article obtained with the worst performing keyword in the article

kw_max_min

Maximum shares any article obtained with the worst performing keyword in the article

kw_avg_min

Average shares any article obtained with the worst performing keyword in the article

kw_min_max

Minimum shares any article obtained with the best performing keyword in the article

kw_max_max

Maximum shares any article obtained with the best performing keyword in the article

kw_avg_max

Average shares any article obtained with the best performing keyword in the article

kw_min_avg

Minimum shares any article obtained with the median performing keyword in the article

kw_max_avg

Maximum shares any article obtained with the median performing keyword in the article

kw_avg_avg

Average shares any article obtained with the median performing keyword in the article

shares

Number of times the article was shared (Target)

Objective

The objective is to predict the number of shares (popularity) on social media platforms for the news articles using the dataset provided. In your submission, you should evaluate at least 3 suitable models for predicting the number of shares.

Overall Evaluation

Once you have identified the most suitable model, find a way to compare the models against each other and explain which of these models is more suitable for the task.

This Mini Project requires you to submit your code!

For this exercise, once you've completed coding in VScode, we'd like you to submit your code to us in GitHub for evaluation and feedback. If you did not attempt this in the previous exercises, the following steps will teach you how to set things up.

Note: Our evaluator uses Python version 3.11. Please ensure that you use Python version 3.11 to ensure package compatibility!

Prerequisites (First-time Git users only)
# Check if Git is installed
git --version

# If not installed, download from https://git-scm.com

# Configure Git with your details (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

Step 1: Create Your Private Repository
Create a new private repository in Github/ GitLab with clear, descriptive naming (e.g., regression-end-to-end)

Instructions for GitHub | Instructions for GitLab

Step 2: Start Coding and Push Your Code
Clone your repository to your local machine:

# For GitHub (use HTTPS URL for simplicity)
git clone https://github.com/yourusername/your-repo-name.git

# Navigate into your project folder
cd your-repo-name

⚠️ Common Issues:

Authentication failed? You may need to use a Personal Access Token instead of password

Repository not found? Check the URL is correct and repository is created

Build your project with the required structure:

root/
├── eda.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
└── main.py

Write your code and complete your project according to the assignment requirements.

Check what you are about to commit:

# See what files will be added
git status
# Review changes in detail (optional)
git diff

Stage your files:

# Add all files
git add .

# OR add specific files only
git add eda.ipynb
git add README.md
git add requirements.txt
git add data/
git add src/
git add main.py

⚠️ Important: Create a .gitignore file to exclude unnecessary files:

# .gitignore content
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
.env
venv/

Commit your changes:

git commit -m "Initial project submission"

💡 Tip: Use descriptive messages like:

"Complete EDA notebook with visualizations"

"Add model training pipeline"

Push to your repository:

# First push needs to set upstream
git push -u origin main
# Future pushes can just use
git push

⚠️ Common Issues:

"main" vs "master"? Check your default branch name with git branch

Push rejected? You might need to pull first: git pull origin main

Large files error? Files over 100MB need Git LFS or should be excluded

Before submitting, ensure:

No sensitive information (passwords, API keys) in code

Data files are under 100MB

All required files are committed

Repository is set to private

You can see your files on GitHub/GitLab web interface

Step 3: Submit Your Work
Once your code is ready in your private repository, click the button below. You'll be guided through creating a Personal Access Token to securely share your repository with us for evaluation.

✅ If we receive your work successfully: We'll evaluate your code and send you detailed feedback via email.

⚠️ If there are issues: We'll let you know what went wrong so you can resubmit.