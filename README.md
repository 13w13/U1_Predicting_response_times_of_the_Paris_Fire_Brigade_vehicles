![ViewCount](https://views.whatilearened.today/views/github/13w13/U1_Predicting_response_times_of_the_Paris_Fire_Brigade_vehicles-.svg?cache=remove)
---
Edgar Jullien, Antoine Settelen, Simon Weiss  
~2 weeks development

Notebook link : https://13w13.github.io/U1_Predicting_response_times_of_the_Paris_Fire_Brigade_vehicles/  
Data Challlenge Provider : https://paris-fire-brigade.github.io/data-challenge/challenge.html
# Introduction

## Presentation of the project

This group project responds to **professor Nadine Galy instructions** written below for U1 - AI&BA - M2 TBS - Econometrics and Statitical Models Project :  
The project involves identifying a real-world business problem or opportunity and designing and implementing an analysis plan to address it using at least one of the modelling methods studied in the course. You are free to choose any business problem or opportunity or public policy issue that you consider challenging and useful to address using business analytics.
The data that you use should be readily available and verifiable.

This is a notebook for the [Paris Fire Brigade](https://paris-fire-brigade.github.io/data-challenge/challenge.html) data challenge 2020 with [ENS](https://challengedata.ens.fr/participants/challenges/21/) and [College de France](https://www.college-de-france.fr/site/stephane-mallat/Challenges-2020.html).


**The goal of this playground challenge** is to predict the *The response times of the Paris Fire Brigade vehicles * which is the delay between:
* the selection of a rescue vehicle (the time when a rescue team is warned) 
* and the rescue team arrival time at the scene of the request (information sent manually via portable radio)

This measurement is composed by the 2 following periods of time: 
* the activation period of the rescue team 
* the transit time of the rescue team

Based on features like trip coordinates, pickup date, type of the arrivall destination, vehicules etc.. 

The [data](https://challengedata.ens.fr/participants/challenges/21/) which covers the entier year 2018 for which inoperable data have been squeezed out comes in the shape of 219 337 training observations and 108 033 test observation. The dataset covers the entire year of 2018. 
Each row contains one Paris fire brigade intervention.

"Response time is one of the most important factors for firefighters because their ability to save lives and rescue people depends on it. Every fire department in the world seeks strategies to decrease their response time, and several analyses have been conducted in the past years to determine what could impact response time. In the meantime, fire departments have been collecting data on their interventions; yet, few of them actually use data science to develop a data-driven decision making approach."https://medium.com/crim/predicting-the-response-times-of-firefighters-using-data-science-da79f6965f93


"A lot of fire departments and emergency services rely on geographic information systems tools, such as ESRI ARCGis or Network Analyst, to obtain estimations about the response time. These tools rely on computing the shortest route using a graphical representation of the road network, which usually gives an accurate estimate of the travel time. Their drawback is that they cannot always take into consideration external dynamic factors such as the weather, traffic or type of units or intervention. Hence, there is an opportunity for machine learning tools to be used here."



**In this notebook**, we will first study and visualize the original data, engineer new features, and examine potential outliers. Then, we implement a boosted Tree for our first model, do some dimension reductions on qualitative features and implement a linear regression. Finaly, we created a final predict and uploaded it to the data plateforme. 

We hope that this notebook will have good results to the challenge and responds fully to Nadine Galy requirement.
As always, any feedback, questions, or constructive criticism are much appreciated.


## Features description


**Input parameters (x_train.csv and x_test.csv):**

* **[ID]** `emergency vehicle selection`: identifier of the selection instance of an emergency vehicle for an intervention
* Intervention
    * `intervention`: identifier of the intervention
    * Alert reason
        * `alert reason category` (category): alert reason category
        * `alert reason` (category): alert reason
    * Address
        * `intervention on public roads` (boolean): 1 when it concerns an intervention on public roads, 0 otherwise
        * `floor` (int): floor of the intervention
        * `location of the event` (category): qualifies the location of the emergency request, for example: entrance hall, boiler room, motorway, etc.
        * `longitude intervention` (float): approximate longitude of the intervention address. **ATTENTION: `intervention_longitude`** !
        * `latitude intervention` (float): approximate latitude of the intervention address. **ATTENTION: `intervention_latitude`** !
    * Emergency vehicle
        * `emergency vehicle`: identifier of the emergency vehicle
            * `emergency vehicle type` (category): type of the emergency vehicle
            * `rescue center` (category): identifier of the rescue center to which belong the vehicle (parking spot of the emergency vehicle)
        * `selection time` (datetime): selection time of the emergency vehicle
            * `date key selection` (int): selection date in YYYYMMDD format
            * `time key selection` (int): selection time in HHMMSS format
        * State of the emergency vehicle preceding its selection for an intervention
            * Operational status of the vehicle preceding its selection
                * `status preceding selection` (category): status of the emergency vehicle prior to selection. An emergency vehicle is in various statuses during an intervention:
                    * **Selection** - selection of the emergency vehicle by the rescue commitment application
                    * **Departed** - the vehicle starts its route to the location of the emergency request
                    * **Presented** - the vehicle arrives at the location of the request
                    * **Hospital transportation** - the vehicle starts its transport of a victim to hospital
                    * **Hospital arrival** - the vehicle arrives at the hospital
                    * **Leaving hospital** - the vehicle leaves the hospital
                    * **Returned** - the vehicle has returned to its parking spot
                    * **Leave the premises** - because the vehicle can also simply leave the scene of an intervention without having to transport any victim
                    * **Not available** - for various reasons the vehicle can be in an unavailable position
                    * **Not relevant** - statutes without interest
                * `delta status preceding selection-selection` (int): number of seconds before the vehicle was selected when its previous status was entered
            * `departed from its rescue center` (boolean) : 1 when the vehicle departed from its rescue center (emergency vehicle parking spot), 0 otherwise
            * GPS position of the vehicle before departure
                * `longitude before departure` (float): longitude of the position of the vehicle preceding his departure. **ATTENTION: `departure_longitude`** !
                * `latitude previous departure` (float): latitude of the position of the vehicle preceding his departure. **ATTENTION: `departure_latitude`** !
                * `delta position gps previous departure-departure` (int): number of seconds before the selection of the vehicle where its GPS position was recorded (when not parked at its emergency center)
            * GPS tracks
                * `GPS tracks departure-presentation` (float pair list): successive GPS positions (*longitude,latitude;longitude,latitude,* etc.) of the vehicle between departure and presentation. This information is for informational purposes to study vehicle behaviors. (The beacons, emitting the GPS positions of vehicles, are currently not always lit)
                * `GPS tracks departure-presentation datetime` (datetime list): datetime associated with successive GPS positions between the departure and the presentation of the vehicle.
            * Estimated route
                * `OSRM estimated route` (json object): service route response of an OSRM instance (http://project-osrm.org/docs/v5.15.2/api/#route-service) setup with the Ile-de-France OpenStreetMap data
                * `OSRM estimated distance` (float): distance calculated by the OSRM route service
                * `OSRM estimated duration` (float): transit delay calculated by the OSRM route service

**Output parameters (y_train.csv and y_test.csv):**

* **[ID]** `emergency vehicle selection`: identifier of the selection instance of an emergency vehicle for an intervention
* **[TO PREDICT]** `delta selection-departure`(int): elapsed time in seconds between the selection and the departure of the emergency vehicle
* **[TO PREDICT]** `delta departure-presentation`(int): elapsed time in seconds between the departure of the emergency vehicle and its presentation on the intervention scene
* **[TO PREDICT]** `delta selection-presentation `(int): elapsed time in seconds between the selection of the emergency vehicle and its presentation on the intervention scene (delta selection-departure + delta departure-presentation)



**Supplementary files (x_train_additional_file.csv and x_test_additional_file.csv)**

* **[ID]** `emergency vehicle selection`: identifier of the selection instance of an emergency vehicle for an intervention
* `OSRM estimate from last observed GPS position`(json object): service route response from last observed GPS position of an OSRM instance (http://project-osrm.org/docs/v5.15.2/api/#route-service) setup with the Ile-de-France OpenStreetMap data
* `OSRM estimated distance from last observed GPS position`(float): distance (in meters) calculated by the OSRM route service from last observed GPS position
* `OSRM estimated distance from last observed GPS position`(float): distance (in meters) calculated by the OSRM route service from last observed GPS position
* `OSRM estimated duration from last observed GPS position`(float): transit delay (in seconds) calculated by the OSRM route service from last observed GPS position
* `time elapsed between selection and last observed GPS position` (float): in seconds
* `updated OSRM estimated duration` (float): time elapsed (in seconds) between selection and last observed GPS position + OSRM estimated duration from last observed GPS position


Good reading ! 
