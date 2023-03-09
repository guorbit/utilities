# *Software repository and code template*

The purpose of this repository is to provide a template for software development, by the GU orbit software team. This repository serves both as a template for file structure and code and contains various tutorials and guides for how we will develop software.

## **General tips**

#### **vs code**
Easiest to use vs code to develop code. This is avaiable at https://code.visualstudio.com/

Use the following vs code extensions:
1. Python
2. autoDocstring - Python Docstring Generator

## **Python Notes**
### **Requirements to contribute to GU Orbit sotware team**
1. Always use a virtual enviornment to develop python code, see below for how to set this up.
2. Always add docstrings to all functions
3. Store and share code on GU Orbit github repositories
4. use a personal git branch, and use pull requests to merge to main branch

### **General Python Commands**
It may be necessary to replace  'python' with 'python3', 'py', or 'py3' depending on your system. 
To run a python script via the command line use command:
>python 'script_name'.py

To install a package to the virtual environment use command:
>python -m pip install 'package_name'

To create a requirements folder use command:
>python -m pip freeze > requirements.txt 

To install a packages specfied in a file called requiremnts.txt use command:
>python -m pip install -r requirements.txt

To run unittests use command:


### **Python Virtual Environments**

To create a virtual envrionment called venv_name (typically just called "venv") use command:
>python -m venv 'venv_name'

To activate the virtual environment located in folder venv_name on windows cmd use command:
>'venv_name'\Scripts\activate

To activate the virtual environment located in folder "venv_name" in a bash terminal use command:
>'venv_name'\bin\activate

Once the virtual environment is active the bash or cmd terminal will indicate this by showing (venv_name) at the command prompt. The active virtual enviornment is a normal python environment and you interact with it exactly like a normal python installtion. To install a package to the virtual environment use command is the same as for normal Python (see above).

### **Unittests**
If python code files are located in a 'src' directory and tests are located in a 'tests' directory, then the following command can be used to run all tests in the 'tests' directory:
>python -m unittest discover -s tests

To run a specific test file use command:
>python -m unittest tests.'test_file_name'

## **Git Notes**


### **Tips**
1. If new use vs code github functionality (found under "source control" tab to the left of screen).
2. When installing git for windows make sure you install the git credneitalm anagaer core. This will allow you to store your username and password for github so you don't have to enter it every time you push to github.

### **Setting up Git for the first time**

Once git is installed on your computer, you will need to set up your git account. To do this, open a terminal and type the following commands:

>git config --global user.name "Your Name"
>git config --global user.email "

To get github to save your password you will use the git credential manager, if you are on windows then you can install this while installing git. If you are on linux or mac then you will need to install this seperately. To install this on linux or mac use the following commands:
>brew install git-credential-manager

Followed by:

> git-credential-manager install

### **General Git Commands**
Always update your local copy of a repository to the current github copy using command while located in the directory on the terminal:
>git pull

### **Preparing a local branch to contribute to an existing repository**
First get a local copy of the repository by cloning the existing repository, to do this you require the web url from github e.g. For this repo: https://github.com/guorbit/software_template.git (Note address should end in '.git'). Then use command:

>git clone "https_address"

The second step is to setup a development branch on which you will develop your code, this has 3 stages:

1. create a nwe local branch
2. switch to this local branch
3. push (inform repository of new branch) to github

To  create a new local branch use command below. Note, the quatation marks should not be added, and the branch name shoud have no spaces.
>git branch 'new_branch_name'

To switch to this branch use command:
>git switch 'new_branch_name'

To push this branch to github use command:
>git push -u origin 'new_branch_name'

After this you should be able to see the new branch on the github website in the repository by clicking the branch dropdown menu at the top left.

### **Pushing code to the repository**
The easiest way to push code to a Github repository is to use VS codes inbuilt git functionality. The procedure to push code to git hub has 3 steps:
1. add a files to be commited
2. commit current changes 
3. push all stored commits to github

To add a specfic file called file_name to the current commit use command:
>git add "file_name"

To add all changes to the current commit use command:
>git add .

To commit changes use command below. Note, in this case the quatation marks around the message are mandatory:
>git commit -m "your message"

Finally, to push all stored commits to github use command:
>git push


### **Creating a pull request**
A pull request is a request to merge your current branch into the main/master branch. To create a pull request on github, first push your code to github (see above). Then go to the github website and click on the "pull requests" tab. Then click "new pull request". Then select the branch you want to merge into the main branch. Then click "create pull request". Then add a title and description to the pull request. Then click "create pull request".


