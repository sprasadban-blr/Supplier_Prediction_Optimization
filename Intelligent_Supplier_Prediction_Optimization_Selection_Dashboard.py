import streamlit as st
import altair as alt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pulp as p

# https://discuss.streamlit.io/t/streamlit-restful-app/409/2
# https://stackoverflow.com/questions/68273958/how-would-i-get-my-streamlit-application-to-use-a-flask-api-in-order-to-retrieve
# https://docs.streamlit.io/

@st.cache
def load_data():
    # Load data
    dataDf = pd.read_csv('Supplier Charateristics.csv', ",")
    return dataDf

def getModel(dataDf):
    X = dataDf.drop(columns=['Quality', 'Delivery', 'Service'],axis=1)
    y = dataDf[['Quality', 'Delivery', 'Service']]    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    
    model = DecisionTreeRegressor()
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force the scores to be positive
    n_scores = absolute(n_scores)
    # summarize performance
    print('Mean and Std Dev: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    model.fit(train_x, train_y)
    predict_train = model.predict(train_x)
    print('Target on train data', predict_train[:5]) 

    mas = mean_absolute_error(train_y, predict_train)
    print('Mean Absolute Error on train dataset : ', mas)
    mse = math.sqrt(mean_squared_error(train_y, predict_train))
    print('Mean Squared Error on train dataset : ', mse)    
    return (model, mas, mse)

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Current Price',
        tooltip=['Current Price', 'Quality', 'Delivery', 'Service']
    ).interactive()
    st.write(graph)

def getOptimalValues(exp_price, exp_quality, exp_delivery, exp_service):    
    # Read Supplier Data
    df = pd.read_csv("Supplier_Data.csv")

    # Create LP minimization problem
    problem = p.LpProblem('SupplierSelection', p.LpMinimize)

    # Create Problem Variables
    suplrs = p.LpVariable.dicts("", df['Abbreviation'], cat='Binary')

    # Objective Function
    # Example: 4500*S1+5000*S2+4300*S3+4600*S4+4650*S5
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] * df['Quoted_Price'][i] for i in range(df['Abbreviation'].count()))

    # Constraints
    #1 #Example: S1+S2+S3+S4+S5 = 1 
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] for i in range(df['Abbreviation'].count())) == 1

    #2 # Example: 4500*S1+5000*S2+4300*S3+4600*S4+4650*S5 <= 4700
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] * df['Quoted_Price'][i] for i in range(df['Abbreviation'].count())) <= exp_price

    #3 For Quality
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] * df['Quality'][i] for i in range(df['Abbreviation'].count())) >= exp_quality

    #4 For Delivery
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] * df['Delivery'][i] for i in range(df['Abbreviation'].count())) >= exp_delivery

    #5 For Service
    problem += p.lpSum(suplrs[df['Abbreviation'][i]] * df['Service'][i] for i in range(df['Abbreviation'].count())) >= exp_service

    # Solve problem
    problem.solve()

    # Print status
    status = p.LpStatus[problem.status]
    print("Status:", status)

    # Print optimal values of decision variables
    
    supplier = []
    selectedSupplier = ""
    for v in problem.variables():
        print(v.name, "=", v.varValue)
        supplier.append(v.name.replace("_", "") + " = " + str(v.varValue))
        if v.varValue is not None and v.varValue > 0:
            selectedSupplier = v.name
    quotedPrice = p.value(problem.objective)
    print("Quoted Price : ", quotedPrice)
    
    supplier = "Optimal value for supplier's " + "'" + str(supplier) + "' and the optimized objective function value is " + str(quotedPrice)
    return (problem, supplier, status, quotedPrice, selectedSupplier)

    
def main(dataDf, modelChar):
    model = modelChar[0]
    mas = modelChar[1]
    mse = modelChar[2]
    
    page = st.sidebar.selectbox("Choose a page", ["Supplier Data", "Data Exploration", "Model Metrics", "Supplier Prediction"])

    if page == "Supplier Data":
        st.header("Historical Data On Supplier Characteristics.")
        st.write("Please select a page on the left.")
        st.write(dataDf)
    elif page == "Data Exploration":
        st.title("Supplier Characteristics Data Exploration")
        y_axis = st.selectbox("Choose a variable for the y-axis", dataDf.columns, index=0)        
        x_axis = st.selectbox("Choose a variable for the x-axis", dataDf.columns, index=1)         
        visualize_data(dataDf, x_axis, y_axis)
    elif page == "Model Metrics":
        st.title("Decision Tree Algorithm Metrics")
        st.write("Model with Mean Squared Error (MSE) =", mse)
        st.write("Model with Mean Absolute Error (MAS) =", mas)        
    elif page == "Supplier Prediction":
        st.title("Supplier Prediction, Optimization and Selection")
        price = st.number_input('Maximum Price')
        
        if price > 0:
            inputData = pd.DataFrame([[price]])
            predictedData = model.predict(inputData)

            st.header("The model predicted with following supplier characteristics.")
            quality = round(predictedData[0][0])
            delivery = round(predictedData[0][1])
            service = round(predictedData[0][2])
            chars = 'Quality: ' + str(quality) + ", Delivery: " + str(delivery) + ", Service: " + str(service)
            st.write(chars)

            optimalValues = getOptimalValues(price, quality, delivery, service)

            st.header("Optimize solver status")
            #st.write("Not Solved: Status prior to solving the problem.")
            #st.write("Optimal: An optimal solution has been found.")
            #st.write("Infeasible: There are no feasible solutions (e.g. if you set the constraints x <= 1 and x >=2).")
            #st.write("Unbounded: The constraints are not bounded, maximising the solution will tend towards infinity.")
            #st.write("Undefined: The optimal solution may exist but may not have been found.")
            sdf = pd.read_csv("Supplier_Data.csv")
            st.write(sdf)
            st.markdown(optimalValues[0])
            print(optimalValues[0])
            
            st.header("Supplier selection and its optimal values.")
            st.write(optimalValues[1])
            st.write('Status = ', optimalValues[2])
            if optimalValues[2] == 'Optimal':
                finalResult = "Supplier chosen '"+ optimalValues[4].replace("_", "") + "' with price = " + str(optimalValues[3])
                st.write(finalResult)

#Load data and create ML model
dataDf = load_data()
modelChar = getModel(dataDf)

#Render Data and Predict model
main(dataDf, modelChar)