__author__ = 'tmkasun'

from pyspark.mllib.linalg import Vectors

import operator
import csv


class SparkAnalyser(object):
    def __init__(self, spark_context):
        self._sc = spark_context

    def load_data(self, location):
        """
        Load text file from local disk, Infact it only creating a pointer here, not loading the hole file to the memory
        :param location: String pointing to data files absolute path
        :return: RDD object
        """
        return self._sc.textFile(location)

    @staticmethod
    def data_extractor(line):
        """

        :param line: Data lines read from the input_rdd file

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

    @staticmethod
    def empty_cost_filter(data_set):
        """

        :param data_set: Comma separated values of the input data row
        :return: At-least one should have a value on it
        """
        return len(data_set[24]) != 0 or len(data_set[25]) != 0

    @staticmethod
    def cost_transform_mapper(dataset):
        """
        If the fuel cost per 6000 miles is given convert it to fuel cost per 12000 miles
        :param dataset: Empty cost filtered dataset for input data row
        :return: Transformed value
        """
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

    @staticmethod
    def electric_vehicles_filter(dataset):
        condition = dataset[10] != "Electricity"
        return condition

    @staticmethod
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
        return feature_vector

