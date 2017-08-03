## Spark Application - execute with spark-submit
# Using Spark 2.1.1
# Using Spark Session (Make Context through Session Build)

# K-Means Anomaly Detection

## Imports
# from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Clustering
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
# Note: Have to install numpy in python
from scipy.spatial import distance

## Module Constantscd
APP_NAME = "Log Cluster"
FILE_NAME = "./kdd-data/kddcup.data.txt"

## Closure Functions
def parse(row):
    """
    Parses a row and returns a named tuple.
    """

    row[0]  = datetime.strptime(row[0], DATE_FMT).date()
    row[5]  = datetime.strptime(row[5], TIME_FMT).time()
    row[6]  = float(row[6])
    row[7]  = datetime.strptime(row[7], TIME_FMT).time()
    row[8]  = float(row[8])
    row[9]  = float(row[9])
    row[10] = float(row[10])
    return Flight(*row[:11])

def cast_columns(spark_df, column_list):
    """
        Helper function takes in a spark dataframe and casts all columns in list to integerType
    """
    # TODO: Add try/catch if column doesn't exist
    for column in column_list:
        spark_df = spark_df.withColumn(column,spark_df[column].cast(IntegerType()))
        # spark_df = spark_df.select(column).cast(IntegerType()))
    return spark_df

def dist_to_centroid(feature_vector, model):
    cluster_number = model.predict(feature_vector)
    print(cluster_number)
    centroid = model.clusterCenters(cluster_number)
    print(centroid)
    dist = distance.euclidean(feature_vector,centroid)
    return(dist)



## Main functionality
def main(sparkSession):
    TEST_RUN = True
    k = 100

    ## Load data
    dataset = sparkSession.read.csv(FILE_NAME, header=False)
    print(dataset.columns)
    # print(dataset.count())

    ## Count labels
    category_column = dataset.columns[-1]
    if TEST_RUN:
        dataset = dataset.where(dataset[category_column] == 'teardrop.')
        k=10
    dataset.groupBy(category_column).count().orderBy(desc('count')).show(10)

    ## Remove categorical columns
    columns_to_drop = dataset.columns[1:4]
    # Note: to drop multiple columns, pass comma-separated group (not a list)
    # E.g. dataset = dataset.drop('_c0', '_c1', '_c2', '_c41')
    # Note: Have to reassign to dataframe on drop
    for column in columns_to_drop:
        dataset = dataset.drop(column)
    print(dataset.columns)

    ## Convert Values to Numbers
    columns = dataset.columns
    feature_columns = [column for column in columns if column != category_column]
    print(feature_columns)
    dataset = cast_columns(dataset,feature_columns)
    print(dataset.columns)

    # Drop rows with Null values
    print('Number of null rows: '+str(dataset.count() - dataset.dropna().count()))
    dataset = dataset.dropna()
    # dataset = dataset.na.fill(0)

    ## Vectorize features
    # Transform feature columns into a single vector, outputCols is new column name
    vecAssembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features")
    dataset_with_feature_vec = vecAssembler.transform(dataset)
    cluster_dataset = dataset_with_feature_vec.select(['features'])
    print(cluster_dataset.columns)

    # Run the model
    kmeans = KMeans() \
        .setK(k) \
        .setSeed(1)
    model = kmeans.fit(cluster_dataset)
    #transformed = model.transform(spark_df_features).select('*')

    # Shows the result
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    # Find anommaly amongst original dataset
    threshold = 0.8
    # dist = dist_to_centroid(feature_vector, model)

    row_list = cluster_dataset.select('features').collect()
    feature_vectors = [row.features for row in row_list]
    print(len(feature_vectors))

    x = sparkSession.sparkContext.parallelize(feature_vectors)
    distances = x.map(dist_to_centroid, model)
    # print(len(distances.collect()))

    print(distances)


if __name__ == "__main__":
    # Create Spark Session
    sparkSession = SparkSession.builder \
        .appName(APP_NAME) \
        .getOrCreate()

    # Execute Main functionality
    main(sparkSession)
