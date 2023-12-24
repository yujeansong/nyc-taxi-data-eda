<a name="br1"></a> 

Predicting Hourly Demand for Yellow Taxi Rides in

New York City

Yujean Song

Student ID: 1186495

[Github](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[ ](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[repo](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[ ](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[with](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[ ](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)[commit](https://github.com/MAST30034-Applied-Data-Science/mast30034-project-1-yujeansong)

August 19, 2023

1 Introduction

In major cities, traditional taxi systems often are challenged with a disproportionate ratio of unproductive

miles—those driven without passengers—to productive miles, resulting in extended waiting times.

However, the last decade has witnessed the increase of on-demand vehicle services like Uber, Lyft, Via,

and Juno, replacing the conventional yellow cab model for several compelling reasons. These for-hire

services excel in convenience and accessibility, thereby garnering increased [inﬂuence.\[1\]](#br9)[ ](#br9)This shift is

particularly noticeable among younger generations who are comfortable utilizing mobile applications

for various services. In response, this report conducts a thorough analysis of the implications for New

York City’s yellow taxi drivers, who are dealing with fewer customers because more and more people

are using these on-demand vehicle services.

Within this study, we will utilize the number of pickups per hour as a means to reﬂect the hourly

demand for taxi services in a particular borough. Our predictive models will be constructed using

combined data on total taxi trips, which indicate the demand, along with weather and event data.

The results of our predictions will provide a more comprehensive insight into the overall demand of a

speciﬁc area, thereby enabling us to deduce latent demand.

During this study, we will introduce and evaluate two distinct machine learning methods to measure

their eﬀectiveness in predicting the hourly passenger demand for taxi services in the New York City

(NYC) area. In addition to its potential beneﬁts for traditional taxi operators and drivers.

1\.1 Dataset

The main dataset selected for our study is the TLC Yellow Taxi Trip Record Data, which

is publicly available from the NYC Taxi Limousine Commission. This dataset includes detailed

information about taxi trips, such as when and where passengers were picked up and dropped oﬀ,

the fare amount, and more, for all completed trips in yellow taxis operating in New York Cit[y\[2\].](#br9)

Our goal was to conduct a useful analysis that would be relevant to yellow taxi drivers in the period

after the oﬃcial end of the COVID-19 pandemic. To ensure that our analysis was based on data

unaﬀected by pandemic-related changes, we aimed to gather information from a time when the

pandemic’s impact was minimal. The pandemic in the United States is considered to have taken

place from late January 2020 to early May [2023\[3\].](#br9)[ ](#br9)Therefore, we chose to use data from a period

before the pandemic signiﬁcantly aﬀected the behavior of local residents and tourists in NYC, and

before travel restrictions were put in place. As a result, we selected the time period of February 2019

1



<a name="br2"></a> 

to July 2019 for our analysis and modeling.

Moreover, external data sets include the New York City Weather Data 2019 [\[4\]](#br9)[ ](#br9)published on

Kaggle with multiple contributors and City of New York Parks Special Events Data [\[5\]](#br9)[ ](#br9)that

is free public data published by New York City agencies and other partners. We integrated diverse

information for analysis, all derived from the concurrent timeframe spanning February 2019 to July

2019\. Additionally, we used Yellow Taxi Zones data [\[6\]](#br9)[ ](#br9)to establish correlations between location IDs

and their respective boroughs.

2 Preprocessing

Numerous preprocessing procedures were done on the three data sets to make sure their alignment

with the structure we intended to follow. The following section will elaborate on the inconsistencies

found in each data set and how we employed to address them.

2\.1 Data Wrangling and Feature Selection

2\.1.1 TLC Trip Record Data

Given the extensive size of the TLC yellow taxi trip record data (43,272,363 rows), we assumed there

will be outliers or abnormalities that could possibly interrupt clear analysis. Furthermore, certain

data sets did not align with the speciﬁc framework we were pursuing, so adjustments were made

accordingly. The following procedures were applied to remove inconsistencies and transformations.

• Irrelevant information features were removed, as it did not align with the relevant information

of interest: Vendor ID, Passenger count, Store, and Forward Flag, Ratecode ID, Payment type,

and other fees related columns got excluded.

• Zero time information was excluded for both passenger pick-up and drop-oﬀ times.

• Instances of zero trip distance time were identiﬁed and removed, as they cannot be

interpreted as instances where taxis were used by passengers.

• Uncertain Location IDs were eliminated for both pick-up and drop-oﬀ locations, as the

location ID is a crucial element of the research. These records have Location IDs outside the

predeﬁned range of 1-263.

• Instances of zero total fares were observed and excluded as they do not qualify as actual

taxi usage. Such instances may arise from errors where a passenger attempted to board but

subsequently decided not to proceed with the trip.

• Unusual extreme values in the trip distance, trip duration, and fares were observed in

a minor amount of data. To identify these extreme values as outliers, a statistical technique was

employed. Using a z-score threshold of 3, data points surpassing this threshold were identiﬁed

and subsequently removed. As an illustration, a record indicating a trip duration of only 13

seconds was eliminated based on the z-score threshold, as such a brief duration was deemed

unrealistic and not representative of a valid trip.

Overall, a total of 457,096 entries were excluded (around 10.5%), resulting in 4,281,267 instances

remaining for our modeling and analysis. The attributes utilized exclusively for modeling and analysis

were the pick-up location ID, pick-up time, pick-up date, and total fares at the outset.

2



<a name="br3"></a> 

2\.1.2 New York City Weather Data 2019

The initial dataset comprises 10 distinct features, but during the extraction process, only the date was

obtained in its original format. The extracted features are followed below:

• Date

• Average Temperature (Fahrenheit)

• New Snow Fall (Inch)

• Precipitation (Inch)

We speciﬁcally ﬁltered records from the 1st of February 2019 to the 31st of July 2019, aligning with

the duration of our study period. Within the initial dataset, the attributes of Average Temperature,

New Snowfall, and Precipitation contain a mix of continuous and discrete variables. However, for our

analysis, we transformed these attributes into categorical variables. This modiﬁcation was undertaken

to focus solely on the occurrence of distinct weather conditions, rather than the intensity of weather

conditions.

2\.1.3 City of New York Parks Special Events Data

The complete data set has a total of 12 features. However, the majority of these features (9 in total)

were deemed irrelevant for the speciﬁc purposes and objectives of our modeling endeavor. Leveraging

NYC Open Data’s capabilities, we eﬃciently ﬁltered a subset of features corresponding precisely to

the desired time period through uncomplicated querying. The chosen features are as follows:

• Date and Time

• Borough

Nonetheless, certain entries within the selected attributes contained missing values, prompting their

exclusion from both the modeling and analysis processes. Furthermore, to enhance the versatility of

feature utilization, we split the date and time components into separate attributes.

2\.1.4 Yellow Taxi Zones Data

The initial data included 4 features and the chosen features are as follows since we are only interested

in the relationship between location ID and corresponding borough:

• Location ID

• Borough

Certain Borough values were absent and marked as ”unknown.” Consequently, we undertook the step

of eliminating these missing values from the dataset.

2\.2 Feature Engineering and Data Aggregation

As previously mentioned, concerning the TLC Trip Record Data, we extracted essential attributes

including pick-up dates and corresponding pick-up location IDs. Subsequently, we decomposed the

pick-up date feature into three distinct components: Hour, Date, and Day of the week. The original

pick-up date attribute was presented in a timestamp format encompassing both date and time.

However, as part of our data preprocessing, we segregated it into Hour and Date components.

3



<a name="br4"></a> 

For instance, consider the pick-up date ”2019-02-01 01:07:27.” Following this transformation, we

isolated the Date as ”2019-02-01” and the Hour as ”01,” where the Hour signiﬁes the speciﬁc hour of

the given time. Furthermore, as an augmentation to the feature set, we introduced a new attribute

denoting the day of the week. In the provided example, the Day of the week attribute would be

represented as ”6,” with each numerical value corresponding to a day of the week: 1 for Monday, 2

for Tuesday, 3 for Wednesday, 4 for Thursday, 5 for Friday, 6 for Saturday, and 7 for Sunday.

Utilizing the processed Event Data and Weather Data, we combined the weather data and TLC trip

data based on the date, merging them with the Yellow Taxi Zones data through an inner join on pick-up

location IDs. Subsequently, we retained only the boroughs as a representative location feature. Lastly,

we proceeded to incorporate the event data into the dataset. This inclusion was achieved by joining

the event data based on the hour, date, and borough attributes.

3 Analysis and Geospatial Visualisation

In this section, we examined the distinct attributes of TLC trip record data individually as well as

their relationship between weather data, and event data.

3\.1 Trend of Taxi Rides Demand

Based on our analysis, apart from location, we identiﬁed a robust correlation between taxi ride demand

and both the day of the week and the time. As we can see in table [1,](#br4)[ ](#br4)we did ANOVA between the

day of the week and the total number of trips. The day of the week is a signiﬁcant factor (P-value:

2\.58 × 10<sup>−24</sup>) and it holds signiﬁcant importance in understanding variations in trip numbers. As

depicted in Figure [1a,](#br5)[ ](#br5)there is no substantial disparity across the various days of the week; however,

Saturday stands out with the highest demand for taxi rides (15.4%), while Monday exhibits the lowest

demand (12.3%).

Subsequently, our focus shifted to conﬁrming whether the trend of heightened demand on the most

popular pick-up day of the week translates to increased revenue per trip for the corresponding day,

as illustrated in Figure [1b.](#br5)[ ](#br5)Our analysis revealed that Friday boasts the highest average fare paid

per trip. Although there exists a discernible diﬀerence compared to other weekdays, the distinction is

particularly pronounced on Sunday.

Table 1: ANOVA between the day of week and Total Number of Trips per Day

SS

DF

6

F

Pr(> F)

Day of Week

Total

20\.9 2.85e-24

4342

In Figure [2,](#br5)[ ](#br5)a consistent pattern emerges across all days of the week in terms of total hourly demand.

Notably, there is a signiﬁcant decline in taxi ride demand from 00 am to 5:59 am and again from 5 pm

to 11:59 pm, which aligns with common expectations. During these hours, a substantial portion of New

Yorkers are typically either asleep or winding down for the night. Research indicates that the average

bedtime for New Yorkers is around 12:00 am [\[7\].](#br9)[ ](#br9)Furthermore, our additional investigation revealed

that the peak rush hour in New York City spans from 5 pm to 7 pm. This timeframe corresponds

to the departure for work and the conclusion of the workday [\[8\].](#br9)[ ](#br9)As traﬃc congestion during these

hours is anticipated to be high, individuals often opt for more aﬀordable public transportation or

4



<a name="br5"></a> 

their personal vehicles. Conversely, taxis tend to become pricier due to congestion charges, and fare

calculations are inﬂuenced by travel time with passengers on board.

(a) The Most Popular Day of the Week

(b) Average Fare Gets Paid per Trip

Figure 2: Total Hourly Demand

3\.2 Pick-up Demands Distributions

Based on our analysis, a strong relationship between location and borough becomes evident. As

depicted in Figure [3a,](#br6)[ ](#br6)it is readily apparent that the average daily pick-up demands are concentrated

heavily around two primary areas: the main New York Airport (John F. Kennedy International) and

the Manhattan district.

We aimed to validate whether locations with higher demand correspondingly generate greater revenue

for individual drivers per trip. Nevertheless, as illustrated in Figure [3b,](#br6)[ ](#br6)it becomes apparent that the

entire borough of New York demonstrates relatively comparable revenue ﬁgures per trip across the

board. This observation underscores that heightened demand does not invariably translate to increased

revenue per trip. Moreover, it suggests that the biggest determinant in hourly revenue for drivers would

perhaps lie in the waiting time to ﬁnd a customer - which is proportional to demand. This would be as

5



<a name="br6"></a> 

opposed to for-hire services that change their rate based on the demand of the location. Additionally,

it is noteworthy that trips to Newark Liberty International Airport and LaGuardia Airport yield a

similar revenue outcome. For both Figure [3a](#br6)[ ](#br6)and [3b,](#br6)[ ](#br6)given the data has a skewed distribution, we

generated a plot using the logarithm of the average daily demand. This adjustment was implemented

to normalize the data and facilitate a more insightful comparison.

(a) Total Hourly Demand

(b) Revenue per Trip

3\.3 Weather and Taxi Trip Demands

The signiﬁcance of temperature in inﬂuencing the total daily trip count is evident from Table [2.](#br6)

The study indicates that the presence of rain or snow did not play a statistically signiﬁcant role in

determining the total number of daily trips.

Table 2: ANOVA Temperature and Total Number of Trips per Day

SS

DF

3

F

Pr(> F)

Temperature

Total

23\.3 8.43e-11

4342

4 Modelling

We employed two distinct models, namely Random Forest Classiﬁcation and Support Vector Machine

(SVM), to assess their performance in predicting the hourly demand for taxi rides in New York City.

4\.1 Random Forest Classiﬁcation

Random Forest classiﬁcation is a supervised learning algorithm designed for eﬃcient and accurate

classiﬁcation tasks. [\[9\]](#br9)[ ](#br9)By constructing an ensemble of decision trees based on bootstrapped samples

from the training data, this ensemble approach combats high variance, leading to accurate predictions.

It’s eﬀective for non-linear relationships. Having explored the data, we concluded that the features

and response variables ’Demand per Hour’ don’t have a linear relationship. Hence, we determined

that Random Forest classiﬁcation would be an appropriate choice for our modeling approach.

6



<a name="br7"></a> 

4\.2 Support Vector Machine

SVM is a supervised machine learning algorithm that is useful for classiﬁcation or regression tasks.

SVM operates by identifying the ideal hyperplane that eﬀectively separates data points belonging to

distinct classes within a high-dimensional feature space. [\[10\]](#br9)[ ](#br9)The primary objective of SVM is to

enhance the algorithm’s ability to generalize and make predictions by maximizing the margin between

these classes.

In our model, we operated under the assumption of independence among the categorical variables,

meaning we considered no interactions between them. Our target variable is ”Demand per hour,”

categorized into ﬁve distinct levels: ”Very High,” ”High,” ”Medium,” ”Low,” and ”Very Low.” As a

result, the SVM classiﬁcation involves the creation of ﬁve separate hyperplanes, each responsible for

distinguishing one of these classes.

A separating Hyperplane in D dimensions can be deﬁned by a normal w and an intercept b

Y = Xβ + ϵ

(1)

Example of an aligned equation (& denotes the symbol to align):

w = [w , w , . . . , w ]

1

2

d

The Hyperplane passing a point x

w x +, . . . , w x + b = 0

1

1

D

D

Given the equation:

w⃗ · ⃗x + b = 0

Where:

• w⃗ and ⃗x are column vectors.

• The dot product w⃗ · ⃗x is usually written as w<sup>⊺</sup>x.

• For this modeling, w , w , w , w , w features such as temperature, day of the week, hour,

1

2

presence of an event, and boroughs.

3

4

5

Linear Classiﬁer takes the form f(x) = w<sup>⊺</sup>x + b in multiple dimension spaces.

5 Discussion

Both models were developed with the aim of predicting hourly classiﬁed demand using a range

of attributes. As classiﬁcation models, our assessment of their performance involved utilizing key

evaluation metrics such as accuracy, precision, recall, and F1 scores. To enhance the consistency of

features’ scales within the SVM model, we standardized the data prior to training. A comparison

of the evaluation metrics presented in table [3](#br8)[ ](#br8)reveals that both models exhibit similar performance

trends, albeit with Random Forest displaying a slight edge in performance.

Although we achieved relatively successful results, it has to be noted that our analysis did have a

downfall in that it lacked data analysis for a wide range of locations. This was because the external

weather dataset only had records for a partial number of boroughs in the NYC area and as such we

were limited to analysis over only those said boroughs present.

7



<a name="br8"></a> 

However, this analysis and modeling endeavor is anticipated to arouse the curiosity of yellow taxi

drivers by revealing novel insights. These insights encompass strategic information, such as identifying

the most proﬁtable days of the week, pinpointing optimal timeframes for maximizing driver productivity,

and oﬀering valuable guidance on selecting proﬁtable destinations.

Table 3: Evaluation Metric

SVM RFC

Accuracy 0.84 0.88

Precision 0.83 0.86

Recall

F1-score

0\.83 0.86

0\.82 0.86

6 Recommendations

The outcomes of the model evaluation underscore the fairly good eﬃcacy of both the RFC and SVM

models in capturing the hourly demand for taxi rides in New York City. As a recommendation aimed

at individual yellow taxi drivers, we propose focusing on high-traﬃc zones such as the Manhattan

area and the major airports. As depicted in Figure [3b,](#br6)[ ](#br6)it’s evident that signiﬁcant revenue potential

exists not only at John F. Kennedy International Airport but also at two other prominent airports.

Consequently, we strongly advise drivers to devise routes that encompass airport stops and cover the

Manhattan region. Furthermore, paying attention to the depicted pick-up hours in Figure [2](#br5)[ ](#br5)is crucial,

suggesting the importance of avoiding rush hours to optimize earnings.

Moreover, as a suggestion for yellow taxi companies, we encourage yellow taxi ﬁrms in NYC to

explore the possibility of creating and ﬁne-tuning similar models using the internal datasets that they

exclusively accumulated. By doing so, they can pave the way for an intelligent dispatch system that

leverages demand predictions to optimize taxi allocation. When demand is predicted to be relatively

high, dispatch more taxis to those areas in advance. Conversely, during periods of predicted relatively

low demand, a reduction in available taxis within those areas can be implemented, contributing to a

more eﬃcient allocation strategy. Furthermore, use the model’s predictions to minimize driver idle

time by directing them to locations with anticipated medium to high demand. Drivers can then be

led to these areas, increasing the likelihood of ﬁnding passengers and reducing the time spent waiting

unproductively.

7 Conclusion

This report comprehensively investigates the potential of predictive modeling to optimize New York

City’s yellow taxi services. By employing SVM and RFC, the study explores the prediction of hourly

taxi demand, addressing key challenges faced by traditional taxi systems and presenting strategies

for resource allocation and driver productivity enhancement. The analysis uncovers factors aﬀecting

demand and suggests practical steps for yellow taxi companies and drivers to succeed in a dynamic

transportation environment.

8



<a name="br9"></a> 

References

[1] Christina Majaski. Uber vs. Yellow Cabs in New York City: What’s the Diﬀerence? [https](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[:](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)

[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[www](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[.](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[investopedia](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[.](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[com](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[articles](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[personal](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[-](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[finance](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[021015](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[/](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[uber](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[-](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[versus](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[-](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[yellow](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[ ](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)[-](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp)

[cabs-new-york-city.asp](https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp). (Visited on 02/21/2021).

[2] TLC Trip record data. New York City Taxi and Limousine Commission. [https://www.nyc.](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

[gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Accessed: 2023-08-04.

[3] Pifzer. Global and U.S. Agencies Declare End of COVID-19 Emergency. [https://www.pfizer.](https://www.pfizer.com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:~:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%20on%20May%2011.)

[com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:](https://www.pfizer.com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:~:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%20on%20May%2011.)

[:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%](https://www.pfizer.com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:~:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%20on%20May%2011.)

[~](https://www.pfizer.com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:~:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%20on%20May%2011.)

[20on%20May%2011.](https://www.pfizer.com/news/announcements/global-and-us-agencies-declare-end-covid-19-emergency#:~:text=Following%20an%20initial%20announcement%20earlier,States%2C%20effective%20on%20May%2011.). Accessed: 2023-08-17. (Visited on 05/11/2023).

[4] moth. New York City Weather Data 2019. [https://www.kaggle.com/datasets/alejopaullier/](https://www.kaggle.com/datasets/alejopaullier/new-york-city-weather-data-2019)

[new-york-city-weather-data-2019](https://www.kaggle.com/datasets/alejopaullier/new-york-city-weather-data-2019). Accessed: 2023-08-04.

[5] NYC Open Data. Parks Special Events. [https](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[:](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[/](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[/](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[data](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[.](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[cityofnewyork](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[.](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[us](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[/](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[Recreation](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[ ](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)[/](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data)

[Parks-Special-Events/6v4b-5gp4/data](https://data.cityofnewyork.us/Recreation/Parks-Special-Events/6v4b-5gp4/data). Accessed: 2023-08-06.

[6] NYC Open Data. “New York Taxi Zone Look up.csv”. Semester 1 2023 MAST30034 Tutorial

Data.

[7] Katherine Ripley. How Much Sleep Do New Yorkers Get? Here’s What The Data Says. [https:](https://secretnyc.co/how-much-sleep-do-new-yorkers-get/)

[//secretnyc.co/how-much-sleep-do-new-yorkers-get/](https://secretnyc.co/how-much-sleep-do-new-yorkers-get/). Accessed: 2023-08-18.

[8] Patrick Davis. What Time Is Rush Hour In NYC? [https://tripmemos.com/what-time-is-](https://tripmemos.com/what-time-is-rush-hour-in-nyc/)

[rush-hour-in-nyc/](https://tripmemos.com/what-time-is-rush-hour-in-nyc/). Accessed: 2023-08-18.

[9] Coporate Finance Institute Team. Random Forest. [https://corporatefinanceinstitute.](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.)

[com/resources/data-science/random-forest/#:](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.)[ ](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.)[:text=Among%20all%20the%20available%](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.)

[~](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.)

[20classification,other%20classes%20in%20the%20data.](https://corporatefinanceinstitute.com/resources/data-science/random-forest/#:~:text=Among%20all%20the%20available%20classification,other%20classes%20in%20the%20data.). Accessed: 2023-08-18.

[10] Hasti Samadi. “Support Vector Machine”. Semester 1 2022 COMP30027 Lecture Notes.

9


