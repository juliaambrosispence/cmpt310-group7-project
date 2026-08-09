# CMPT 310 Group 7 Project

## Project Requirements
This project uses `Python 3.12.1` and requires the `pip` package manager to install requirements inside a virtual environment.

## Project Setup
The fastest way to get up to speed is to use [JetBrain's PyCharm IDE ](https://www.jetbrains.com/pycharm/) which should recognize the `requirements.txt`
file and prompt you to set up the virtual environment and interpreter settings.

Alternatively, open the repository in your terminal/command prompt and create a virtual environment with the command

`python -m venv .venv`

Then activate the virtual environment you just created using the script:

`.\.venv\Scripts\Activate.bat` on Windows

`source .venv/bin/activate` on macOS or Linux

Finally run the installation command in the now activated virtual environment using the `requirements.txt` file:

`pip install -r requirements.txt`

Now all you need to ensure is whatever IDE you're using has the .venv directory installation set as the Python interpreter.
Using the virtual environment keeps all the dependencies independent of the state of whatever Python installation a user may already have.

Finally, with the virtual environment active, run
`python src/recipe_classifier/main.py`
to run the project.