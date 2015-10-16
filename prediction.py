__author__ = 'tmkasun'

from pyspark.context import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeansModel

from matplotlib import pyplot
import numpy as np
import operator
import csv

from conf.configurations import project

spark_configuration = SparkConf().setAppName("Vehicles_Cluster_prediction").setMaster('local')
spark_context = SparkContext(conf=spark_configuration)


def load_data(location):
    """
    :param location: String pointing to data files absolute path
    :return: RDD object
    """
    return spark_context.textFile(location)


def data_extractor(line):
    """
        Sample Data Structure for reference:
        Description => file Index => 0  Value = DataPartB_july2000.csv      # *****Removed
        Description => year Index => 1  Value = 2000                        # *****Removed
        Description => manufacturer Index => 2  Value = BMW                 0
        Description => model Index => 3  Value = 3 Series E36               # *****Removed
        Description => description Index => 4  Value = 316i Compact         # *****Removed
        Description => euro_standard Index => 5  Value = 3                  1
        Description => tax_band Index => 6  Value =                         # *****Removed
        Description => transmission Index => 7  Value = A4                  # *****Removed
        Description => transmission_type Index => 8  Value = Automatic      # *****Removed
        Description => engine_capacity Index => 9  Value = 1895             2 # Divisor
        Description => fuel_type Index => 10  Value = Petrol                3
        Description => urban_metric Index => 11  Value = 12                 4 # Multiply
        Description => extra_urban_metric Index => 12  Value = 7.1          5 # Multiply
        Description => combined_metric Index => 13  Value = 8.9             6 # Multiply
        Description => urban_imperial Index => 14  Value = 23.5             7 # Multiply
        Description => extra_urban_imperial Index => 15  Value = 39.8       8 # Multiply
        Description => combined_imperial Index => 16  Value = 31.7          9 # Multiply
        Description => noise_level Index => 17  Value = 72                  10 # Multiply
        Description => co2 Index => 18  Value = 213                         11 # Multiply
        Description => thc_emissions Index => 19  Value = 131               # *****Removed
        Description => co_emissions Index => 20  Value = 1058               12 # Multiply # TODO: need to add for the multiplication
        Description => nox_emissions Index => 21  Value = 63                13 # Multiply # TODO: need to add for the multiplication
        Description => thc_nox_emissions Index => 22  Value =               # *****Removed
        Description => particulates Index => 23  Value =                    # *****Removed
        Description => fuel_cost_6000_miles Index => 24  Value =            14
        Description => fuel_cost_12000_miles Index => 25  Value = 671       15 # Multiply
        Description => date_of_change Index => 26  Value =                  # *****Removed
    """
    csv_list = list(csv.reader([line], delimiter=',', quotechar='"'))[0]
    return csv_list


def empty_cost_filter(data_set):
    return len(data_set[24]) != 0 or len(data_set[25]) != 0


def cost_transform_mapper(dataset):
    new_dataset = dataset[:24]

    fuel_cost_6000_miles = dataset[24]
    if fuel_cost_6000_miles == '':
        fuel_cost_6000_miles = 0.0
    else:
        fuel_cost_6000_miles = float(fuel_cost_6000_miles)

    fuel_cost_12000_miles = dataset[25]
    if fuel_cost_12000_miles == '':
        fuel_cost_12000_miles = 2 * fuel_cost_6000_miles
    else:
        fuel_cost_12000_miles = float(fuel_cost_12000_miles)

    return new_dataset + [fuel_cost_12000_miles]


def electric_vehicles_filter(dataset):
    condition = dataset[10] != "Electricity"
    return condition


def feature_mapper(filtered_dataset):
    euro_std = int(filtered_dataset[5])
    engine_capacity = float(filtered_dataset[9])

    try:
        mul_valued = reduce(operator.mul, map(float, filtered_dataset[11:19])) * filtered_dataset[24]
    except ValueError:
        mul_valued = 0
        euro_std = 0

    aggregated_normalized_feature = mul_valued / engine_capacity
    related_data = [euro_std, aggregated_normalized_feature]

    feature_vector = Vectors.dense(related_data)
    filtered_dataset.append(feature_vector)
    return filtered_dataset


def predict_cluster(feature_vector, estimated_clusters):
    print estimated_clusters.prdict(feature_vector)


def manufactures_mapper(optimum_data):
    return [optimum_data[2], 1]  # ('manufacture',1)


def main():
    data_rdd = load_data(project['data_file'])
    print(data_rdd.count())

    listed_data_rdd = data_rdd.map(data_extractor)

    # Filtering unwanted data rows
    elect_filtered_rdd = listed_data_rdd.filter(electric_vehicles_filter)
    filtered_rdd = elect_filtered_rdd.filter(empty_cost_filter)

    # Mapping related data to convenient format for clustering
    cost_tx_rdd = filtered_rdd.map(cost_transform_mapper)
    feature_mapped_rdd = cost_tx_rdd.map(feature_mapper)

    estimated_clusters = KMeansModel.load(spark_context, "identified_clusters")

    optimum_cluster = 4
    optimum_points_rdd = feature_mapped_rdd.filter(
        lambda filtered_data_feature_vector: estimated_clusters.predict(
            filtered_data_feature_vector[-1]) == optimum_cluster)

    print(optimum_points_rdd.count())

    sample_data = optimum_points_rdd.take(100)

    for data in sample_data:
        feature_vector = data[-1]
        pyplot.scatter(feature_vector[0], feature_vector[1])

    optimum_cluster_manufactures_rad = optimum_points_rdd.map(manufactures_mapper)

    optimum_points_rdd.persist()  # To hold the previously calculated data set in memory

    individual_manufactures_count_rdd = optimum_cluster_manufactures_rad.reduceByKey(operator.add)

    sorted_manufactures_count_rdd = individual_manufactures_count_rdd.sortBy(
        lambda manufactures_set: manufactures_set[1], ascending=False)

    top_ten = 10
    vehicle_count = []
    manufactures_name = []
    for manufacture in sorted_manufactures_count_rdd.take(top_ten):
        vehicle_count.append(manufacture[1])
        manufactures_name.append(manufacture[0])
        print(manufacture)

    pyplot.title("Best Vehicle Cluster")
    pyplot.xlabel("Europe Rating")
    pyplot.ylabel("Feature Normalized")

    pyplot.show()

    number_of_manufactures = top_ten  # sorted_manufactures_count_rdd.count()
    index = np.arange(number_of_manufactures)

    bar_width = 0.5

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    chart = pyplot.bar(index, vehicle_count, bar_width,
                       alpha=opacity,
                       color='b',
                       error_kw=error_config,
                       label='manufactures')
    pyplot.xticks(index + bar_width, manufactures_name)

    pyplot.title("Top preforming vehicles")
    pyplot.xlabel("manufactures")
    pyplot.ylabel("Vehicles count")

    pyplot.show()


if __name__ == '__main__':
    main()