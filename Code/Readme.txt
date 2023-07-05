1. To open notebook 
    python3 -m notebook

Steps to run the code
1. first download the code from github.
2. Extract the code in your local repository.
3. Install python in local.
4. Install virtualenv library
    windows:
    pip install virtualenv
    mac:
    pip3 install virtualenv
5. Open terminal, navigate to Code folder and create virtual environment
    virtualenv myenv
6. Activate the virtual environment
    windows:
    myenv/Scripts/Activate
    mac:
    source myenv/bin/Activate
7. Install all the requirement libraries.
    pip3 install -r requirements.txt
8. Open Notebook.ipynb in jupyter and run the code to create the required models.(specificallu SVM)
9. run the python application.
    flask run
10. access the application through http://localhost:5000