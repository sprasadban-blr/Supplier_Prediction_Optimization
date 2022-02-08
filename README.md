# Supplier characteristics Prediction, Finding Optimized Supplier and Supplier Selection
POC to check concept on Supplier Prediction, Optimization and Supplier selection process

Prerequisite:
  * Install Python 3.6 https://www.python.org/downloads/release/python-368/ 
	* Install needed python libraries
		- python -m pip install --upgrade pip
		- pip install sklearn
		- pip install --upgrade streamlit
		- pip install pulp
		- pip install altair
		- pip install matplotlib
		- pip install pandas
		- pip install numpy
		
Step 1: Clone git
  * $SRC_DIR>git clone https://github.com/sprasadban-blr/Supplier_Prediction_Optimization

Step 2: Run dashboard application 
  * $SRC_DIR>streamlit run Intelligent_Supplier_Prediction_Optimization_Selection_Dashboard.py

Step 3: Run UI application from browser
  * http://localhost:8501/

Highlights:
  * Performing supplier characteristics prediction using multi output regression 
  * Selecting one among the best regression algorithms (Liner Regression, KNN, Decision Tree, Direct Multioutput Regression, Chained Multioutput Regression)
  * Deriving optimization problem with set of linear equations based on supplier characteristics
  * Choosing best supplier based on predicted supplier characteristics and executing optimization linear equations.
