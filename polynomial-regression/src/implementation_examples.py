# Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from functions import LLS_Solve
from functions import LLS_ridge
from functions import poly_func
from functions import RMSE
# ------------------------------------------------------------------------------


# Linear Regression with Athens Temperature Data
# ------------------------------------------------------------------------------

# Example: implementing the solution to the least squares problem.

# Solve the least squares linear regression problem for the Athens
# temperature data.  Make sure to annotate the plot with the RMSE.

# read athens data into pandas dataframe
athens_dataframe = pd.read_csv('../datasets/athens_ww2_weather.csv')
# min temp will be our input data 'x'
# get min temp from athens dataframe
min_temp = athens_dataframe['MinTemp']
# max temp will be our output data 'y'
# get max temp from athens dataframe
max_temp = athens_dataframe['MaxTemp']

# choose degree of polynomial to be rmse value
# we choose 1 since this is a linear function (or should be)
degree = 1

# calculate w for this degree
w = LLS_Solve(min_temp, max_temp, degree)

# find rmse
rmse = RMSE(min_temp, max_temp, w)

# calculate polynomial output vector
polynomial_output_vector = poly_func(min_temp, w)

# initialize figure
fig1, ax = plt.subplots()

# plot original athens data as scatter plot
ax.plot(min_temp, max_temp, 'o', label="original data")
# plot fitted line
ax.plot(min_temp, polynomial_output_vector, label="linear regression model")

# set title and axes
ax.set_title('Athens Temperature')
ax.set_xlabel('Min Temperature')
ax.set_ylabel('Max Temperature')

# annotate plot with rmse
ax.annotate("rmse: {}".format(rmse), xy=(9.25, 37.5), xycoords='data',
            arrowprops=dict(facecolor='black', shrink=0.5),
            horizontalalignment='right', verticalalignment='top', bbox=dict(boxstyle="square,pad=0.3", fc="white", lw=2))

plt.legend()


# Polynomial Regression with the Yosemite Visitor Data
# ------------------------------------------------------------------------------

# Create degree-n polynomial fits for 5 years of the Yosemite data, with
# n ranging from 1 to 20.  Additionally, create plots comparing the
# training error and RMSE for 3 years of data selected at random (distinct
# from the years used for training).

# read yosemite data csv into pandas dataframe
yosemite_data = pd.read_csv('../datasets/Yosemite_Visits.csv')

# get 5 most recent years
five_years = yosemite_data.head(5).drop('Year', 1)

# initialize list to hold months in integer representation
# and add each instance of each month over these 5 years to this list
input_data = []
month_ints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
for x in month_ints:
    for y in range(0, 5):
        input_data.append(x)

# convert list to np array
input_data = np.array(input_data)

# get month names
months = five_years.columns.values

# initialize list to hold visitors for each corresponding month in input_data
datapoints = []

# populate datapoints
for month in months:
    for year in range(0, 5):
        data = five_years.iloc[year][month]
        data = data.replace(',', '')
        data = int(data)
        datapoints.append(data)

# convert datapoints to np array
datapoints = np.array(datapoints)

# plot graphs for degrees ranging from 1 - 20
for deg in range(1, 21):
    plt.figure(figsize=(12.5, 8))
    # plot separate scatter for each year
    for year in range(0, 5):
        year_specific_data = five_years.iloc[year].to_numpy()
        clean_year_specific_data = []
        for datapoint in year_specific_data:
            datapoint = datapoint.replace(',', '')
            datapoint = int(datapoint)
            clean_year_specific_data.append(datapoint)
        final_year_specific_data = np.array(clean_year_specific_data)
        plt.plot(month_ints, final_year_specific_data, '*', label='year: {}'.format(yosemite_data.iloc[year]['Year']))
    # calculate w for this degree
    w = LLS_Solve(input_data, datapoints, deg)
    # calculate polynomial output vector
    polynomial_output_vector = poly_func(input_data, w)

    # plot the fitted vector
    plt.plot(input_data, polynomial_output_vector, label="fitted line for degree: {}".format(deg))

    # set title and axes labels
    plt.title("Yosemite Visitors for degree: {}".format(deg))
    plt.xlabel("Month")
    plt.ylabel("Visitors")
    plt.xticks(month_ints, months, fontsize=5)

    plt.legend()


# ------------------------------------------------------------------------------

# get list of years not including five years used above
years = yosemite_data['Year'].to_numpy()
years = years[5:]

# convert to numpy integer array
clean_years = []
for year in years:
    year = int(year)
    clean_years.append(year)

years = np.array(clean_years)

# initialize list to hold random ints
random_years = []
# populate random_years list
while len(random_years) != 3:
    random_year = random.choice(years)
    if random_year not in random_years:
        random_year = int(random_year)
        random_years.append(random_year)

random_years = np.array(random_years)

initial_year = yosemite_data.iloc[0]['Year']

select_data = yosemite_data.drop('Year', 1)

for year in random_years:
    data = select_data.loc[initial_year - year].to_numpy()
    clean_data = []
    for datapoint in data:
        datapoint = datapoint.replace(',', '')
        datapoint = int(datapoint)
        clean_data.append(datapoint)
    clean_data = np.array(clean_data)

    degrees_vector = []
    rmse_vector = []
    training_error_vector = []
    for degree in range(1, 21):
        degrees_vector.append(degree)
        coeffs = LLS_Solve(input_data, datapoints, degree)
        polynomial_output_vector = poly_func(input_data, coeffs)
        training_error = RMSE(input_data, datapoints, coeffs)
        training_error_vector.append(training_error)
        w = LLS_Solve(month_ints, clean_data, degree)
        rmse = RMSE(month_ints, clean_data, w)
        rmse_vector.append(rmse)
    degrees_vector = np.array(degrees_vector)
    training_error_vector = np.array(training_error_vector)
    rmse_vector = np.array(rmse_vector)

    plt.figure(figsize=(12, 8))
    plt.plot(degrees_vector, rmse_vector, label="year")
    plt.plot(degrees_vector, training_error_vector, label="training error")
    plt.title("Comparing Training Error for Fitted Data vs. RMSE for Year {}".format(year))
    plt.xlabel("Month")
    plt.ylabel("Visitors")
    plt.xticks(month_ints, months, fontsize=5)

    plt.legend()


# Solve the ridge regression regularization fitting for 5 years of data for
# a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
# values from 0 to 1.  Annotate the plots with this value.

# fix degree to n = 10
degree = 10

# get spacing
spacing = 1/20

lam = 0

for deg in range(1, 21):
    plt.figure(figsize=(12.5, 8))
    # plot separate scatter for each year
    for year in range(0, 5):
        year_specific_data = five_years.iloc[year].to_numpy()
        clean_year_specific_data = []
        for datapoint in year_specific_data:
            datapoint = datapoint.replace(',', '')
            datapoint = int(datapoint)
            clean_year_specific_data.append(datapoint)
        final_year_specific_data = np.array(clean_year_specific_data)
        plt.plot(month_ints, final_year_specific_data, '*', label='year: {}'.format(yosemite_data.iloc[year]['Year']))
    # calculate w for this degree for ridge regression problem
    w = LLS_ridge(input_data, datapoints, degree, lam)
    # calculate polynomial output vector
    polynomial_output_vector = poly_func(input_data, w)

    lam = round((lam + spacing), 2)

    # plot the fitted vector
    plt.plot(input_data, polynomial_output_vector, label="ridge regression fitted line for lambda: {}".format(lam))

    # set title and axes labels
    plt.title("Yosemite Visitors for degree: {}".format(degree))
    plt.xlabel("Month")
    plt.ylabel("Visitors")
    plt.xticks(month_ints, months, fontsize=5)

    plt.legend()

plt.show()