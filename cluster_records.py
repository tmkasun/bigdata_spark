__author__ = 'tmkasun'

from pyspark.context import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans

from matplotlib import pyplot

from conf.configurations import project
from libs.analyser import SparkAnalyser


def main():

    # Setup Spark context by setting application name and running mode, `localhost` is a special string here
    spark_configuration = SparkConf().setAppName("Vehicles_KMean_clustering").setMaster('local')
    spark_context = SparkContext(conf=spark_configuration)

    s_analyser = SparkAnalyser(spark_context)

    data_rdd = s_analyser.load_data(project['data_file'])

    # TODO: Only for debug purpose
    # init_data_count = data_rdd.count()
    # print("INFO: Number of records in the data set {}".format(init_data_count))

    listed_data_rdd = data_rdd.map(s_analyser.data_extractor)

    # Filtering unwanted data rows
    elect_filtered_rdd = listed_data_rdd.filter(s_analyser.electric_vehicles_filter)
    filtered_rdd = elect_filtered_rdd.filter(s_analyser.empty_cost_filter)

    # Mapping related data to convenient format for clustering
    cost_tx_rdd = filtered_rdd.map(s_analyser.cost_transform_mapper)
    feature_mapped_rdd = cost_tx_rdd.map(s_analyser.feature_mapper)

    # TODO: Only for debug purpose
    debug_message = "INFO: Filtered Data count = {} Rejected count = {}\nSample line `{}`"
    filtered_data_count = cost_tx_rdd.count()
    # rejected_data = init_data_count - filtered_data_count
    # print(debug_message.format(filtered_data_count, rejected_data))

    iterations_count = 10
    runs_count = 10
    k_value = 5
    model = KMeans.train(feature_mapped_rdd, k_value, iterations_count, runs_count)
    wssse = model.computeCost(feature_mapped_rdd)
    print("INFO: Within Set Sum of Squared Error = {}".format(wssse))

    cluster_centers = model.clusterCenters
    model.save(spark_context, "identified_clusters")

    for center in cluster_centers:
        pyplot.scatter(center[0], center[1])

    pyplot.title("Cluster Centroids")
    pyplot.xlabel("Europe Label")
    pyplot.ylabel("Feature Normalized")
    pyplot.show()


if __name__ == '__main__':
    main()