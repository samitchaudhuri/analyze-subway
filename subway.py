#import sys
import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas
import pandasql
import csv
import scipy
import scipy.stats
from random import sample
import linreg
from ggplot import *

#my.ggsave <- function(filename = default_name(plot), height = 5, width = 4, dpi = 72, ...) {
#    ggsave(filename='plot/'+filename, height=height, width=width, dpi=dpi, ...)

# SQL queries on subway data
#----------------------------
# Use SQL query to count the number of rainy days from a dataframe of weather data from
# https://www.dropbox.com/s/7sf0yqc9ykpq3w8/weather_underground.csv
def num_rainy_days(filename):
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT
    COUNT(*)
    FROM weather_data
    WHERE rain = '1';
    """
    #Execute your SQL command against the pandas frame
    rainy_days = pandasql.sqldf(q.lower(), locals())
    return rainy_days

# Use SQL query to return two columns and two rows - whether it was foggy or not (0
# or 1) and the max maxtempi for that fog value (i.e., the maximum
# max temperature for both foggy and non-foggy days).
def max_temp_aggregate_by_fog(filename):
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT fog, max(cast (maxtempi as integer))
    FROM weather_data
    GROUP BY fog;
    """
    #Execute your SQL command against the pandas frame
    foggy_days = pandasql.sqldf(q.lower(), locals())
    return foggy_days

# Use SQL query to compyt average mean temperature on weekends
def avg_mintemp_weekend(filename):
    q = """
    SELECT AVG(cast (meantempi as integer))
    FROM weather_data
    WHERE cast(strftime('%w', date) as integer) = 0 or cast(strftime('%w', date) as integer) = 6;
    """
    #Execute your SQL command against the pandas frame
    mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
    return mean_temp_weekends

# Use SQL query to compyt average minimum temperature on rainy days
def avg_mintemp_rainy(filename):
    weather_data = pandas.read_csv(filename)

    q = """
    SELECT AVG(cast (mintempi as integer))
    FROM weather_data
    WHERE rain = '1' and cast(mintempi as integer) > 55;
    """    
    
    #Execute your SQL command against the pandas frame
    mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
    return mean_temp_weekends

# Investigate underground weather with SQL queries on subway data
def check_underground_weather(weather_filename):
    output_filename = "rainy_days.csv"
    rainy_days_df = num_rainy_days(weather_filename)
    rainy_days_df.to_csv(output_filename)

    output_filename = "maxtemp_by_fog.csv"
    maxtemp_fog_df = max_temp_aggregate_by_fog(weather_filename)
    maxtemp_fog_df.to_csv(output_filename)

    output_filename = "avg_mintemp_weekend.csv"
    mintemp_weekend_df = avg_mintemp_weekend(weather_filename)
    mintemp_weekend_df.to_csv(output_filename)

    output_filename = "avg_mintemp_rainy.csv"
    mintemp_rainy_df = avg_mintemp_rainy(weather_filename)
    mintemp_rainy_df.to_csv(output_filename)


# Wrangle subway data
# ----------------------------------------------
def fix_turnstile_data(dataset_path, filenames):
    '''
    Filenames is a list of MTA Subway turnstile text files. A link to an example
    MTA Subway turnstile text file can be seen at the URL below:
    http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt
    
    As you can see, there are numerous data points included in each row of the
    a MTA Subway turnstile text file. 

    You want to write a function that will update each row in the text
    file so there is only one entry per row. A few examples below:
    A002,R051,02-00-00,05-28-11,00:00:00,REGULAR,003178521,001100739
    A002,R051,02-00-00,05-28-11,04:00:00,REGULAR,003178541,001100746
    A002,R051,02-00-00,05-28-11,08:00:00,REGULAR,003178559,001100775
    
    Write the updates to a different text file in the format of "updated_" + filename.
    For example:
        1) if you read in a text file called "turnstile_110521.txt"
        2) you should write the updated data to "updated_turnstile_110521.txt"

    The order of the fields should be preserved. 
    
    You can see a sample of the turnstile text file that's passed into this function
    and the the corresponding updated file in the links below:
    
    Sample input file:
    https://www.dropbox.com/s/mpin5zv4hgrx244/turnstile_110528.txt
    Sample updated file:
    https://www.dropbox.com/s/074xbgio4c39b7h/solution_turnstile_110528.txt
    '''
    for inName in filenames:
        infilename = dataset_path+inName
        with open(infilename, 'rb') as inFile:
            outfilename = "updated"_inName
            with open(outfilename, 'wb') as outFile:
                csvReader = csv.reader(inFile, delimiter=',')
                csvWriter = csv.writer(outFile, delimiter=',')
                for row in csvReader:
                    for idx in range(8):
                        newRow = row[0:3] + row[3+5*idx:8+5*idx]
                        csvWriter.writerow(newRow)
                outFile.close()
            inFile.close()

def create_master_turnstile_file(filenames, output_file):
    '''
    Write a function that takes the files in the list filenames, which all have the 
    columns 'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn', and consolidates
    them into one file located at output_file.  There should be ONE row with the column
    headers, located at the top of the file.
    
    For example, if file_1 has:
    'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
    line 1 ...
    line 2 ...
    
    and another file, file_2 has:
    'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
    line 3 ...
    line 4 ...
    line 5 ...
    
    We need to combine file_1 and file_2 into a master_file like below:
     'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
    line 1 ...
    line 2 ...
    line 3 ...
    line 4 ...
    line 5 ...
    '''
    with open(output_file, 'w') as master_file:
       master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
       csvWriter = csv.writer(master_file, delimiter=',')
       for inName in filenames:
           with open(inName, 'rb') as inFile:
               csvReader = csv.reader(inFile, delimiter=',')
               csvReader.next()
               for row in csvReader:
                   csvWriter.writerow(row)
               inFile.close()
       master_file.close()

def filter_by_regular(filename):
    '''
    This function should read the csv file located at filename into a pandas dataframe,
    and filter the dataframe to only rows where the 'DESCn' column has the value 'REGULAR'.
    
    For example, if the pandas dataframe is as follows:
    ,C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn
    0,A002,R051,02-00-00,05-01-11,00:00:00,REGULAR,3144312,1088151
    1,A002,R051,02-00-00,05-01-11,04:00:00,DOOR,3144335,1088159
    2,A002,R051,02-00-00,05-01-11,08:00:00,REGULAR,3144353,1088177
    3,A002,R051,02-00-00,05-01-11,12:00:00,DOOR,3144424,1088231
    
    The dataframe will look like below after filtering to only rows where DESCn column
    has the value 'REGULAR':
    0,A002,R051,02-00-00,05-01-11,00:00:00,REGULAR,3144312,1088151
    2,A002,R051,02-00-00,05-01-11,08:00:00,REGULAR,3144353,1088177
    '''
    turnstile_data = pandas.read_csv(filename)
    turnstile_data = turnstile_data[turnstile_data.DESCn == 'REGULAR']
    return turnstile_data

def get_hourly_entries(df):
    '''
    The data in the MTA Subway Turnstile data reports on the cumulative
    number of entries and exits per row.  Assume that you have a dataframe
    called df that contains only the rows for a particular turnstile machine
    (i.e., unique SCP, C/A, and UNIT).  This function should change
    these cumulative entry numbers to a count of entries since the last reading
    (i.e., entries since the last row in the dataframe).
    
    More specifically, you want to do two things:
       1) Create a new column called ENTRIESn_hourly
       2) Assign to the column the difference between ENTRIESn of the current row 
          and the previous row. If there is any NaN, fill/replace it with 1.
    
    You may find the pandas functions shift() and fillna() to be helpful in this exercise.
    
    Examples of what your dataframe should look like at the end of this exercise:
    
           C/A  UNIT       SCP     DATEn     TIMEn    DESCn  ENTRIESn    EXITSn  ENTRIESn_hourly
    0     A002  R051  02-00-00  05-01-11  00:00:00  REGULAR   3144312   1088151                1
    1     A002  R051  02-00-00  05-01-11  04:00:00  REGULAR   3144335   1088159               23
    2     A002  R051  02-00-00  05-01-11  08:00:00  REGULAR   3144353   1088177               18
    3     A002  R051  02-00-00  05-01-11  12:00:00  REGULAR   3144424   1088231               71
    4     A002  R051  02-00-00  05-01-11  16:00:00  REGULAR   3144594   1088275              170
    5     A002  R051  02-00-00  05-01-11  20:00:00  REGULAR   3144808   1088317              214
    6     A002  R051  02-00-00  05-02-11  00:00:00  REGULAR   3144895   1088328               87
    7     A002  R051  02-00-00  05-02-11  04:00:00  REGULAR   3144905   1088331               10
    8     A002  R051  02-00-00  05-02-11  08:00:00  REGULAR   3144941   1088420               36
    9     A002  R051  02-00-00  05-02-11  12:00:00  REGULAR   3145094   1088753              153
    10    A002  R051  02-00-00  05-02-11  16:00:00  REGULAR   3145337   1088823              243
    ...
    ...

    '''
    hourlyEntries = df.ENTRIESn - df.ENTRIESn.shift(1) 
    df['ENTRIESn_hourly'] = hourlyEntries.fillna(1)
    return df

def get_hourly_exits(df):
    '''
    The data in the MTA Subway Turnstile data reports on the cumulative
    number of entries and exits per row.  Assume that you have a dataframe
    called df that contains only the rows for a particular turnstile machine
    (i.e., unique SCP, C/A, and UNIT).  This function should change
    these cumulative exit numbers to a count of exits since the last reading
    (i.e., exits since the last row in the dataframe).
    
    More specifically, you want to do two things:
       1) Create a new column called EXITSn_hourly
       2) Assign to the column the difference between EXITSn of the current row 
          and the previous row. If there is any NaN, fill/replace it with 0.
    
    You may find the pandas functions shift() and fillna() to be helpful in this exercise.
    
    Example dataframe below:

          Unnamed: 0   C/A  UNIT       SCP     DATEn     TIMEn    DESCn  ENTRIESn    EXITSn  ENTRIESn_hourly  EXITSn_hourly
    0              0  A002  R051  02-00-00  05-01-11  00:00:00  REGULAR   3144312   1088151                0              0
    1              1  A002  R051  02-00-00  05-01-11  04:00:00  REGULAR   3144335   1088159               23              8
    2              2  A002  R051  02-00-00  05-01-11  08:00:00  REGULAR   3144353   1088177               18             18
    3              3  A002  R051  02-00-00  05-01-11  12:00:00  REGULAR   3144424   1088231               71             54
    4              4  A002  R051  02-00-00  05-01-11  16:00:00  REGULAR   3144594   1088275              170             44
    5              5  A002  R051  02-00-00  05-01-11  20:00:00  REGULAR   3144808   1088317              214             42
    6              6  A002  R051  02-00-00  05-02-11  00:00:00  REGULAR   3144895   1088328               87             11
    7              7  A002  R051  02-00-00  05-02-11  04:00:00  REGULAR   3144905   1088331               10              3
    8              8  A002  R051  02-00-00  05-02-11  08:00:00  REGULAR   3144941   1088420               36             89
    9              9  A002  R051  02-00-00  05-02-11  12:00:00  REGULAR   3145094   1088753              153            333
    '''
    
    hourlyExits = df.EXITSn - df.EXITSn.shift(1) 
    df['EXITSn_hourly'] = hourlyExits.fillna(0)
    return df


def time_to_hour(time):
    '''
    Given an input variable time that represents time in the format of:
    "00:00:00" (hour:minutes:seconds)
    
    Write a function to extract the hour part from the input variable time
    and return it as an integer. For example:
        1) if hour is 00, your code should return 0
        2) if hour is 01, your code should return 1
        3) if hour is 21, your code should return 21
        
    Please return hour as an integer.
    '''
    
    dt = pandas.to_datetime(time, dayfirst=True)
    hour = dt.hour
    return hour


def reformat_subway_dates(date):
    '''
    The dates in our subway data are formatted in the format month-day-year.
    The dates in our weather underground data are formatted year-month-day.
    In order to join these two data sets together, we'll want the dates formatted
    the same way.  
    More info can be seen here:
    http://docs.python.org/2/library/datetime.html#datetime.datetime.strptime
    '''
    dt = datetime.datetime.strptime(date, "%m-%d-%y")
    date_formatted = dt.strftime("%Y-%m-%d")
    return date_formatted

def fix_master_data(turnstile_weather):
    # Disable false warnings from potetial chained assignment
    pandas.options.mode.chained_assignment = None

    # add new column datetime
    turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']

    # rename column 'Hour' to 'hour'
    turnstile_weather.rename(columns = {'Hour':'hour'}, inplace=True)
    #turnstile_df['hour'] = turnstile_df['TIMEn'].map(lambda x: pandas.to_datetime(x, dayfirst=True).hour)

    # add new columns day_week,weekday
    turnstile_weather['day_week'] = turnstile_weather['datetime'].map(lambda x: pandas.to_datetime(x, dayfirst=True).weekday())
    turnstile_weather['weekday'] = turnstile_weather['day_week'].map(lambda x: 1 if x < 5 else 0)

    #pandasql does not deal well with strangely named columns
    turnstile_weather = turnstile_weather.rename(columns={'Unnamed: 0':'Idx_audit'})
    return turnstile_weather

def wrangle_turnstile_data(dataset_path, turnstile_files, master_data_filename, final_data_filename):
    fix_turnstile_data(dataset_path, turnstile_files)
    create_master_turnstile_file(turnstile_files, master_data_filename)

    # Filter irregular data
    turnstile_df = filter_by_regular(master_data_filename)
    turnstile_df.to_csv('turnstile_data_regular.csv')

    # Add hourly entries
    turnstile_df = turnstile_df.groupby(['C/A','UNIT','SCP']).apply(
        get_hourly_entries)
    turnstile_df.to_csv('turnstile_data_hourly_entries.csv')

    # Add get hourly exits
    #turnstile_df = pandas.read_csv('turnstile_data_hourly_entries.csv')
    turnstile_df = turnstile_df.groupby(['C/A','UNIT','SCP']).apply(
        get_hourly_exits)
    turnstile_df.to_csv('turnstile_data_hourly_entries_exits.csv')

    # Convert time to hour
    #turnstile_df = pandas.read_csv('turnstile_data_hourly_entries_exits.csv')
    turnstile_df['Hour'] = turnstile_df['TIMEn'].map(time_to_hour)
    turnstile_df.to_csv('turnstile_data_hour.csv')

    # Reformat dates
    #turnstile_df = pandas.read_csv('turnstile_data_master_subset_time_to_hour.csv')
    turnstile_df['DATEn'] = turnstile_df['DATEn'].map(reformat_subway_dates)
    turnstile_df.to_csv(final_data_filename)

def examine_master_data(df, useImprovedData):
    # Check the period of time-series data
    unitName = 'R003'
    date = '05-01-11' if (useImprovedData == True) else '2011-05-01'
    hours = df[(df.UNIT==unitName) & (df.DATEn==date)].TIMEn
    print "Unit ", unitName, " has logged entries on ", date, " at hours:"
    print hours.values

    imagename='unithist_improved.png' if (useImprovedData == True) \
        else 'unithist.png'
    filtered_df = df[(df.UNIT=='R003') | (df.UNIT=='R550')][['UNIT', 'hour']]
    filtered_df.UNIT = filtered_df['UNIT'].astype('category')
    title = 'Histogram of Hour'
    if (useImprovedData==True):
        title += ' (Improved Data)'
    unit_plot = ggplot(aes(x='hour', color='UNIT', fill='UNIT'), data=filtered_df) +\
        geom_histogram(binwidth=1) +\
        ggtitle(title) + \
        xlab('Hour or Entry') + ylab('Number of Records')
    ggsave(imagename, unit_plot, width=11, height=8)


    # Plot number of records per hour
    imagename='hrhist_improved.png' if (useImprovedData==True) else 'hrhist.png'
    title = 'Histogram of Hour' 
    if (useImprovedData==True):
        title += ' (Improved Data)'
    rph_plot = ggplot(aes(x='hour'), data=df) +\
        geom_histogram(binwidth=1) +\
        ggtitle(title) +\
        xlab('Hour or Entry') + ylab('Number of Records')
    ggsave(imagename, rph_plot, path='assets/images', width=6, height=4, bbox_inches='tight')


# Visualize Subway Data
# ------------------------------------
def entries_histogram(turnstile_weather, useImprovedData):
    '''
    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    
    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    # plot a historgram for hourly entries when it is raining and not raining - use matplotlib
    #plt.figure()
    #turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly'].hist(range=(0,6000), bins = 20, label=['No Rain'])
    #turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly'].hist(range=(0,6000), bins = 20, label=['Rain']) 
    #plt.xlabel('Entriesn_hourly')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of ENTRIESn_hourly')
    #plt.legend()
    #return plt

    # plot a historgram for hourly entries when it is raining and not
    # raining - use ggplot
    plot = ggplot(aes(x='ENTRIESn_hourly', fill='rain', color='rain',
                      legend=True), data=turnstile_weather) +\
           geom_histogram(binwidth=300) +\
           xlim(low=0, high=6000) +\
           ggtitle('Histogram of ENTRIESn_hourly') +\
           xlab('Entriesn_hourly') +\
           ylab('Frequency')
    return plot

def examine_residuals(outcomes, predictions, prefix):
    # http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    # Test residuals for normal distribution
    residuals = outcomes - predictions
    (zscore, pvalue) = scipy.stats.normaltest(residuals)
    legLabel = 'mean = {0}, pvalue = {1}'.format(residuals.mean(), pvalue)
    print prefix, "residual", legLabel

    # Plot a histogram of residuals
    plt.figure()
    residuals.hist(bins=20, label=legLabel)
    plt.xlabel(prefix+" Residuals")
    plt.ylabel("Frequency")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    return plt

def plot_riders_density(turnstile_weather):
    plot = ggplot(turnstile_weather, aes(x='ENTRIESn_hourly', color='rain')) + \
        geom_density() + \
        xlim(0, 3000) +\
        facet_wrap('weekday') +\
        labs(x='Entriesn_hourly', y='Count of Records')
    return plot

def plot_riders_by_hour(turnstile_weather):
    # Ridership by time of day 
    df = turnstile_weather[['hour', 'rain', 'ENTRIESn_hourly']]
    df = df.groupby(['hour', 'rain'], as_index=False).aggregate(np.sum)
    plot = ggplot(df, aes(x='hour', y='ENTRIESn_hourly', color='rain', fill='rain')) + \
        geom_line(position = 'stack') + \
        ggtitle('Total Number of Entries in each Hour of the Day.') +\
        labs(x='Hour', y='Total Number of Entries (over all Stations)')
    return plot

def plot_riders_by_day(turnstile_weather):
    # Ridership by day of week
    df = turnstile_weather[['day_week', 'rain', 'ENTRIESn_hourly']]
    df = df.groupby(['day_week', 'rain'], as_index=False).aggregate(np.sum)
    plot = ggplot(df, aes(x='day_week', y='ENTRIESn_hourly', fill='rain', colour='rain')) + \
        geom_line(position = 'stack') +\
        ggtitle('Total Number of Entries in each Day of the Week.') +\
        labs(x='Day', y='Total Number of Entries (over all Stations)')
    return plot

def plot_riders_by_station(turnstile_weather):
    df = turnstile_weather[['UNIT', 'latitude', 'longitude', 'ENTRIESn_hourly']]
    #df['station_index'] = df['UNIT'].map(lambda x: int(x[1:]))
    #df = df[hour_df.station_index > 15]
    #df = df[hour_df.station_index < 26]
    #df['station_index'] = df['UNIT'].astype('category')
    df_sum = df.groupby(['UNIT'], as_index=False)['ENTRIESn_hourly'].aggregate(np.sum)
    df = df.groupby(['UNIT'], as_index=False).aggregate(np.max)
    df.ENTRIESn_hourly = df_sum.ENTRIESn_hourly
    station_plot = ggplot(aes(x='longitude', y='latitude',
                              size='ENTRIESn_hourly',
                              saturation='ENTRIESn_hourly'), data=df) +\
        geom_point() +\
        ggtitle("MTA Entries By Station") + xlab('Station') + ylab('Entries')
    return station_plot

# Which stations have more exits or entries at different times of day
def plot_busiest_stations(turnstile_weather):
    df = turnstile_weather[['hour', 'latitude', 'ENTRIESn_hourly', 'EXITSn_hourly']]

    idx = df.groupby(['hour'])['ENTRIESn_hourly'].transform(max)==df['ENTRIESn_hourly']
    df_entries = df[idx].sort(['hour']).reset_index()
    df_melt_entries = pandas.melt(df_entries, id_vars=['hour', 'latitude'], value_vars = ['ENTRIESn_hourly'])

    idx = df.groupby(['hour'])['EXITSn_hourly'].transform(max)==df['EXITSn_hourly']
    df_exits = df[idx].sort(['hour']).reset_index()
    df_melt_exits = pandas.melt(df_exits, id_vars=['hour', 'latitude'], value_vars = ['EXITSn_hourly'])

    df_melt = df_melt_entries.append(df_melt_exits, ignore_index=True)

    plot = ggplot(aes(x='hour', y='latitude', color = 'variable', size='value'), data=df_melt)+\
        geom_point() +\
        ggtitle("Busiest Subway Stations Hour by Hour") + xlab('Hour of the Day') + ylab('Location (Latitude) of Station')

    return plot

# Master data is at https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
# ggplot info: https://pypi.python.org/pypi/ggplot/
def plot_master_weather_data(turnstile_weather, useImprovedData):
    if useImprovedData == True:
        # How ridership varies based on Subway station
        imagename = "riders_bystation_improved.png" if (useImprovedData==True) else 'riders_bystation.png'
        station_plot = plot_riders_by_station(turnstile_weather)
        ggsave(imagename, station_plot, path='assets/images', width=6, height=4, bbox_inches='tight')

        # Which stations have more exits or entries at different times of day
        imagename = "busiest_stations_improved.png" if (useImprovedData==True) else 'busiest_stations.png'
        busiest_stations_plot = plot_busiest_stations(turnstile_weather)
        print busiest_stations_plot
        #ggsave(imagename, busiest_stations_plot, path='assets/images', width=12, height=10)
        ggsave(imagename, busiest_stations_plot, path='assets/images')        
    else:
        # Ridership by hour of the day
        imagename = "riders_byhour_improved.png" if (useImprovedData==True) else 'riders_byhour.png'
        hour_plot = plot_riders_by_hour(turnstile_weather)
        ggsave(imagename, hour_plot, path='assets/images', width=6, height=4, bbox_inches='tight')

        # Ridership by day of week
        imagename = "riders_byday_improved.png" if (useImprovedData==True) else 'riders_byday.png'
        day_plot = plot_riders_by_day(turnstile_weather)
        ggsave(imagename, day_plot, path='assets/images', width=6, height=4, bbox_inches='tight')

        # Rider desity on week days and weekends
        imagename = "riders_density_improved.png" if (useImprovedData==True) else 'riders_density.png'
        density_plot = plot_riders_density(turnstile_weather)
        ggsave(imagename, density_plot, path='assets/images', width=6, height=4, bbox_inches='tight')


# Analyze Subway Data
# ------------------------------------
def run_stats_inference(turnstile_weather):
    '''
    Useful web sites 
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    '''
    
    ### YOUR CODE HERE ###
    with_rain = turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly']
    without_rain = turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly']

    with_rain_mean = np.mean(with_rain)
    without_rain_mean = np.mean(without_rain)
    U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)

    with_rain = turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly']
    without_rain = turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly']
    print "rainy day mean    = ", np.mean(with_rain)
    print "non-rainy day mean = ", np.mean(without_rain)
    U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)
    print "Mann Whitney U = ", U
    print "one-tailed p-value: P(x > y) = ", p
    alpha = 0.5   # significance level
    # one-tailed test, two-tailed hypothesis
    if (2*p) < alpha:
        print "Reject null hypotheis because P (x > y) = {0} < {1}%".format(2*p*100, alpha)
    else:
        print "Cannot reject null hypotheis becuase P(x > y) = {0} >= {1}%".format(2*p*100, alpha)
    #return with_rain_mean, without_rain_mean, U, p # leave this line for the grader


def select_features_and_outcomes(df, aggrDaily):
    #features = df[['weekday','maxpressurei','maxdewpti','mindewpti','minpressurei','meandewpti','meanpressurei','fog','rain','meanwindspdi','mintempi','meantempi','maxtempi','precipi']] 
    if aggrDaily == True:
        outcomes = df.groupby(['DATEn','UNIT'], as_index=False)['ENTRIESn_hourly'].aggregate(np.sum)['ENTRIESn_hourly']
        df = df.groupby(['DATEn','UNIT'], as_index=False)[['rain', 'fog', 'weekday', 'hour']].aggregate(np.max)
    else:
        outcomes = df['ENTRIESn_hourly']

    features = df[['hour', 'rain', 'fog', 'weekday']]

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(df['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    # Normalize features for run time
    features, mu, sigma = linreg.normalize_features(features)
    features['ones'] = np.ones(len(outcomes)) # Add a column of 1s (y intercept)
    print features.shape
    return outcomes, features

def predict_with_gradient_descent(features, outcomes, alpha, num_iterations):
    # Convert features and outcomes to numpy arrays
    features_array = np.array(features)
    outcomes_array = np.array(outcomes)

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = linreg.gradient_descent(
        features_array, outcomes_array, theta_gradient_descent, alpha, 
        num_iterations)
    #print theta_gradient_descent
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions

"""
Instead of using Gradient Descent to compute the coefficients theta
used for the ridership prediction, once can also try using a different reference
implementation such as:
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html

One of the advantages of the statsmodels implementation is that it
gives easy access to the values of the coefficients theta. This can
help to infer relationships between variables in the dataset.

We can also experiment with polynomial terms as part of the input variables.  
The following links might be useful: 
http://en.wikipedia.org/wiki/Ordinary_least_squares
http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
http://en.wikipedia.org/wiki/Polynomial_regression
"""

def predict_with_ols(features, outcomes):
    #
    # Your implementation goes here. Feel free to write additional
    # helper functions
    # 

    m = len(outcomes)

    features, mu, sigma = linreg.normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and outcomes to numpy arrays
    features_array = np.array(features)
    outcomes_array = np.array(outcomes)

    olsmodel = sm.OLS(outcomes_array, features_array)
    olsres = olsmodel.fit()
    #print olsres.params
    predictions = olsres.predict(features_array)
    return predictions

      
if __name__ == '__main__':

    # Configuration variables
    wrangleData = False
    useImprovedData = True
    aggrDaily = True
    visualizeData = True
    infer_stats = True
    runGradientDescent = True
    runOLS = True

    # Read data into a pandas dataframe
    #turnstile_weather = pandas.read_csv(input_filename)
    #turnstile_weather['UNIT'] = turnstile_weather['UNIT'].astype('category')
    # Take a random subset of the data
    #rindices = np.array(sample(xrange(len(turnstile_weather)), 100))
    #turnstile_subset = turnstile_weather.ix[rindices]
    #turnstile_subset.reset_index()
    #print turnstile_subset.describe()

    # Lesson 2 - Wrangle 
    # --------------------------------------------
    dataset_path = './subway_data/'

    # Investigate underground weather with SQL queries on subway data
    if wrangleData:
        weather_filename = dataset_path + 'weather_underground.csv'
        check_underground_weather(weather_filename)

        # Fix and combine turnstile data text files into one master csv file
        turnstile_files = ['turnstile_110528.txt', 'turnstile_110604.txt']
        master_data_filename = "turnstile_data_master.csv"
        final_data_filename = "turnstile_data_final.csv"    
        wrangle_turnstile_data(dataset_path, turnstile_files, master_data_filename, final_data_filename)

    if useImprovedData == True:
        input_filename = dataset_path + "turnstile_weather_v2.csv"
        turnstile_weather = pandas.read_csv(input_filename)
    else:
        input_filename = dataset_path + "turnstile_data_master_with_weather.csv"
        turnstile_weather = pandas.read_csv(input_filename)
        turnstile_weather = fix_master_data(turnstile_weather)
    #examine_master_data(turnstile_weather, useImprovedData)

    # Lesson 4 - Visualize
    # --------------------------------------------
    # Perform Exploratory data analysis. 
    imagename = "entries_hist_improved.png" if (useImprovedData==True) else 'entries_hist.png'
    if visualizeData==True:
        plot = entries_histogram(turnstile_weather, useImprovedData)
        #plot.savefig('assets/images/'+imagename)
        ggsave(imagename, plot, path='assets/images', width=6, height=4,
               bbox_inches='tight')
        plot_master_weather_data(turnstile_weather, useImprovedData)

    # Lesson 3 - Analyze
    # ----------------------------------------
    # The hourly entries data are not normally distributed. So we cannot perfom
    # Welch's T-test. Let's perform mann whitney test.
    if infer_stats == True:
        run_stats_inference(turnstile_weather)

    # Predict with gradient descent linear regression
    outcomes, features = select_features_and_outcomes(turnstile_weather, aggrDaily)
    if runGradientDescent == True:
        # Set values for alpha, number of iterations.
        alpha = 0.1 
        num_iterations = 75 
        gdes_predictions = predict_with_gradient_descent(features, outcomes, alpha, num_iterations)

        # Plot cost_history
        imagename='gdes_cost_history_improved.png' if (useImprovedData==True) \
            else 'gdes_cost_history.png'
        #plot = linreg.plot_cost_history(alpha, cost_history)
        #ggsave(imagename, plot, path='assets/images', width=7, height=5)
    
        # Calculate r-squared
        gdes_r_squared = linreg.compute_r_squared(outcomes, 
                                                  gdes_predictions)
        print "gdes r-squared = ", gdes_r_squared

        # Plot residuals
        imagename='gdes_residuals_improved.png' if (useImprovedData == True) \
            else 'gdes_residuals.png'
        plot = examine_residuals(outcomes, gdes_predictions, "Gradient Descent")
        plot.savefig(imagename, width=7, height=5)

    # Predict with Ordinary Least Squares linear regression
    if runOLS == True:
        ols_predictions = predict_with_ols(features, outcomes)

        # Calculate r-squared
        ols_r_squared = linreg.compute_r_squared(outcomes, ols_predictions) 
        print "ols r-squared = ", ols_r_squared

        #Plot residuals
        imagename='ols_residuals_improved.png' if (useImprovedData == True) \
            else 'ols_residuals.png'
        plot = examine_residuals(outcomes, ols_predictions, "OLS")
        plot.savefig(imagename)

    exit

