#
#  Web Server Log Analysis with Apache Spark.
#
#  Dataset used:
#      Data set from NASA Kennedy Space Center web server in Florida.
#           The full data set is freely available at
#           http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html, and it
#           contains all HTTP requests for two months.
#
#           We are using a subset that only contains several days worth
#           of requests.

"""

Run with:
    spark-submit log_analyzer.py

"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import udf, sum, col
from pyspark.sql import functions as F
from pyspark.sql.functions import dayofmonth
from pyspark.sql.functions import countDistinct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#  -- Parse the log file ---------------------------------------------------- #
def regrex_alog_file(alog_df):
    """ Use Common Log Format """
    regrex_alog_data_frame = alog_df.select(
        regexp_extract(
            'value',
            r'^([^\s]+\s)',
            1).alias('Host'),
        regexp_extract(
            'value',
            r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]',
            1).alias('Timestamp'),
        regexp_extract(
            'value',
            r'^.*"\w+\s+([^\s]+)\s+HTTP.*"',
            1).alias('Path'),
        regexp_extract(
            'value',
            r'^.*"\s+([^\s]+)',
            1).cast('integer').alias('Status'),
        regexp_extract(
            'value',
            r'^.*\s+(\d+)$',
            1).cast('integer').alias('Content_size'))
    return regrex_alog_data_frame


#  -- Cleans the log file-----------------------------------------------------#
def clean_alog_file(palog_df):
    """
    Replace all content values values with 0.
    A UDF to translate the Apache timestamps to "standard" timestamps.
    """
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7,
        'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def count_null(col_name):
        """ Build up a list of column expressions, one per column. """
        return sum(col(col_name).isNull().cast("integer")).alias(col_name)

    exprs = []
    for col_name in palog_df.columns:
        exprs.append(count_null(col_name))

    #print("Total number of null records found in the palog_df:")

    #-- Run the aggregation. The *exprs converts
    #-- the list of expressions into
    #-- variable function argumts.
    # palog_df.agg(*exprs).show()

    #-- Replace all null content_size values with 0.
    nonull_alog_df = palog_df.na.fill({"content_size": 0})
    # nonull_alog_df.count()

    #print("Total number of null records found in the clean DF:")

    exprs = []
    for col_name in nonull_alog_df.columns:
        exprs.append(count_null(col_name))
    # nonull_alog_df.agg(*exprs).show()

    #--Parse the log file timestamp
    def parse_alog_time(custom_time_stamp):
        """
        Convert Common Log time format into a Python datetime object

        Args:
            custom_time_stamp (str): date and time in Apache time
            format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]

        Returns:
            A string suitable for passing to CAST("timestamp")

        """
        return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
            int(custom_time_stamp[7:11]),
            month_map[custom_time_stamp[3:6]],
            int(custom_time_stamp[0:2]),
            int(custom_time_stamp[12:14]),
            int(custom_time_stamp[15:17]),
            int(custom_time_stamp[18:20])
        )

    #--  Now append column to our parsed, cleaned dataframe
    parse_time = udf(parse_alog_time)
    clean_alog_df = nonull_alog_df.select(
        "*",
        parse_time(
            nonull_alog_df["timestamp"]).cast("timestamp")
        .alias("Time")).drop("timestamp")

    return clean_alog_df


#  -- Perform exploratory Data Analysis on Apache log file--------------------#
def eda_alog_file(calog_df):
    """ Calculate statistics based on the content size. """
    calog_df.describe(["content_size"]).show()


#  -- HTTP Status Analysis----------------------------------------------------#
def http_status_alog(calog_df):
    """ Find frequent Status and Graphing  """
    status_count_df = (calog_df.groupBy("Status").count().sort("Status"))
    # status_count_df.show()

    #-- Let"s take the log since the 200 status dominates the count
    log_status_df = status_count_df.withColumn("lCount",
                                               F.log(status_count_df["count"]))
    print("<<<< HTTP Status >>>>")
    log_status_df.show()

    plt.style.use("fivethirtyeight")
    plt_df = log_status_df.toPandas()
    plt_df.plot(x="Status", y="lCount", color="#00a79c", kind="bar", rot=45)
    plt.xlabel("Year")
    plt.ylabel("log(count)")
    plt.title("HTTP Status Analysis")

    plt.show()


#  -- Find out most frquently accessed hosts----------------------------------#
def freq_hosts_alog(calog_df):
    """ Get any hosts that has accessed the server more than 10 times """
    freq_hosts_df = (calog_df.groupBy("Host").count())
    freq_hosts_t10_df = (freq_hosts_df
                         .filter(freq_hosts_df["count"] > 10)
                         .select(freq_hosts_df["host"]))

    print("<<<< Top Ten Hosts >>>>")
    freq_hosts_t10_df.show(10, truncate=False)


#  -- Number of hits to paths (URIs) in the log-------------------------------#
def hits_path_alog(calog_df):
    """ Find out most frequently accessed pages """
    paths_df = (calog_df
                .groupBy("Path")
                .count()
                .sort("count", ascending=False))

    print("<<<< Top ten URIs >>>>")
    paths_df.show(10, truncate=False)

    # paths_hits_df = (paths_df
    #                 .select("Path", "count")
    #                 .rdd.map(lambda r: (r[0], r[1]))
    #                 .collect())

    plt_df = paths_df.toPandas()

    plt.style.use("fivethirtyeight")
    plt_df.plot(x="Path", y="count", color="#00a79c", kind="line", linewidth=5)
    plt.xlabel("Paths")
    plt.ylabel("Number of Hits")
    plt.show()

#  -- Error paths (URIs) in the log-------------------------------------------#
def error_path_alog(calog_df):
    """ DataFrame containing all accesses that did not return a code 200. """
    status_not_eto200 = (calog_df.filter(calog_df['status'] != 200))
    status_not_eto200.show(5)

    status_group_data = status_not_eto200.groupBy("path")
    #-- Once we have the GroupedData object we should be able to run
    #-- aggregations, count the paths, and return a list

    status_gd_paths = status_group_data.agg({"path": "count"}).collect()
    #-- Turn the list into a dataframe and change the column name created
    #-- above from "count(path)" to just "count" by providing a schema list:
    status_gd_pathdf = alog_spark_session.createDataFrame(status_gd_paths,
                                                          schema=["path",
                                                                  "count"])
    #-- status_gd_pathdf is a now a dataframe with columns path and count,
    #-- but not yet sorted
    logs_sum_df = status_gd_pathdf.sort("count", ascending=False)

    print("<<<< Top Ten failed URIs >>>>")
    logs_sum_df.show(10, False)

#  -- Unique hosts in the log-------------------------------------------------#
def unique_host_file(calog_df):
    """ List the unique hosts from the entire log """
    #-- Number of unique hosts
    unique_host_count = calog_df.agg(countDistinct("host").alias(
        "unique")).collect().pop().__getitem__("unique")
    print("Unique hosts: {0:d}".format(unique_host_count))

    #-- Number of Unique Hosts by Day
    day_to_host_pair_df = calog_df.select("host", dayofmonth("time")
                                          .alias("day")).orderBy("day")
    # day_group_hosts_df is the same as above, but with dupes removed:
    day_group_hosts_df = day_to_host_pair_df.dropDuplicates()

    # daily_hosts_df is a Dataframe with two columns; day of month, and
    # count of unique hosts to hit website on that day of month:
    daily_hosts_df = day_group_hosts_df.groupBy("day").count()

    print("Unique hosts per day:")
    daily_hosts_df.show(30, False)
    daily_hosts_df.persist()


if __name__ == "__main__":

    alog_spark_session = SparkSession \
        .builder \
        .appName("Apache log analyzer") \
        .master("local[*]") \
        .getOrCreate()

    apache_log_path = "file:///home/hduser/python/aug_access_log"

    #-- Load and Parse the Apache log file
    apache_log_df = alog_spark_session.read.text(apache_log_path)
    parsed_alog_df = regrex_alog_file(apache_log_df)

    #-- Clean Dataframe to process further data analysis
    cleaned_alog_df = clean_alog_file(parsed_alog_df)

    print("<<<< Apache Log SChema >>>>")
    cleaned_alog_df.printSchema()
    cleaned_alog_df.show(5, truncate=False)

    cleaned_alog_df.cache()

    #-- Calculate statistics based on the content size
    eda_alog_file(cleaned_alog_df)

    #-- HTTP Status Analysis with graph
    http_status_alog(cleaned_alog_df)

    #-- Top Ten Hosts
    freq_hosts_alog(cleaned_alog_df)

    #-- Top Ten URIs/commonly accessed pages
    hits_path_alog(cleaned_alog_df)

    #-- Top Ten failed URIs
    error_path_alog(cleaned_alog_df)

    #-- Unique hosts in the log per day
    unique_host_file(cleaned_alog_df)

    #-- Close the Spark Session gracefully.
    alog_spark_session.stop()
