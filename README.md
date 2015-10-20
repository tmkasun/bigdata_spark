#Steps to re-run the code. 

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
