Necessary data to do the processing was collected from the following site. http://datahub.io/dataset/car-fuel-consumptions-and-emissions/resource/fb1a4163-8ab3-4c88-a791-2fb14209bbce . It contains data of different car models ranging from year 2000 to 2013. Before the data was used to find the intended solution it had to be preprocessed in order to make it consistent and processable. 

##Preprocessing the data
1.  Remove data of cars that run in electricity
2.  Fill the null values with the average values of the column
3.  Remove column with nominal values

There are some columns with values that has comma within the value, which ‘Spark’ blindly breaks it down into columns rather than taking it as one value. Therefore the number of columns has to be taken into account and from rows that has 32 columns and not 31 have to be identified and extra column should be removed. 
	The preprocessing were done using multiple Mapreduce passes with the most efficient way to do most number of operations within one pass. The unique way Spark functions once the process of one request is done it doesn’t write the data back to the distributed file system which enables to continue with the preprocessing one after the other until the final necessary data set is derived. 
	
##Clustering
The clustering generally happens based on the two dimensions we assigned. To reduce the complexity and the dimensions we have created the combined metric which is associated with the engine. The values related to the engine are multiplies which includes the running costs and related fuel consumption and emissions. To make the data more sensible and fair the combined multiplies values are divided by the engine capacity so that the values which naturally increase with the engine size is normalized per engine capacity unit. This makes the analysis very sensible. The other dimension of the clustering is the safety rating which comes from the Euro standards which means it should comply and pass the expected standards in various safety aspect both physically and technologically. This includes multi directional collision test, pedestrian safety, number of airbags effectiveness of crumple zones and so on. So the cluster gives a very sensible information when we consider the lowest combined matrix against the highest safety rating. 
The number of cluster points were decided based on the number of data we have which is around 45000 which divided into 5 clusters would give around 9000 results per cluster. Given the number of manufacturers available this number per cluster would give a us a fair value to create the result we need.

##Steps to re-run the code. 

1. Install Python dependencies:
  Numpy
  Matplotlib
  PySpark


2. Locate `~/.bash_profile`  and append the below lines which will set SPARK_HOME environment variable to locate the spark library files for use in python interpreter 

`export SPARK_HOME="path/to/spark/extracted/directory/spark-1.5.1-bin-hadoop2.6"`
`export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH`
`export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH`

3. Use the data file given here , since we have preprocessed the dataset to remove some wildcard UNICODE character. 
4. Download the source code and goto config directory
5. Change the configuration parameters as necessary
    Set data_file location to, csv data file
    Set sample_data_file if you wish to run on small set of data set to evaluation
