from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import PolynomialExpansion


def main():
    spark = SparkSession.builder \
        .appName('test task') \
		.master('local[*]') \
		.getOrCreate()
    
    # Загрузка данных в датафрейм с заданной схемой
    path_read = 'file:///home/prophet/conda/yakovlev/1t_data/data/yellow_tripdata_2020-01.csv'
    path_write = '/home/prophet/conda/yakovlev/1t_data/result/res_table.parquet'
    path_for_sample = '/home/prophet/conda/yakovlev/1t_data/result/sample_df.csv'

    schema = StructType([
        StructField("VendorID", StringType(), True),
        StructField("tpep_pickup_datetime", TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_distance", FloatType(), True),
        StructField("RatecodeID", StringType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("PULocationID", StringType(), True),
        StructField("DOLocationID", StringType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", FloatType(), True),
        StructField("extra", FloatType(), True),
        StructField("mta_tax", FloatType(), True),
        StructField("tip_amount", FloatType(), True),
        StructField("tolls_amount", FloatType(), True),
        StructField("improvement_surcharge", FloatType(), True),
        StructField("total_amount", FloatType(), True),
        StructField("congestion_surcharge", FloatType(), True)
    ])
    df_taxi = spark.read.load(path_read, format='csv', infer_schema='true', header='true', sep=',', schema=schema)
    # Очистка данных
    df_taxi = df_taxi \
        .filter(F.col('passenger_count').isNotNull()) \
        .withColumn('passenger_count_new', F.when(F.col('passenger_count') >= 5, 5).otherwise(F.col('passenger_count'))) \
        .withColumn('date', F.to_date(F.col('tpep_pickup_datetime')))
    
    # Определение самой дорогой и самой дешевой поездки для каждого дня в разрезе групп по количеству пассажиров
    df_min_max_amount = df_taxi \
        .filter(F.col('total_amount') > 0) \
        .groupby('date', 'passenger_count_new').agg(F.round(F.min('total_amount'), 2).alias('min_total_amount'),
                                                    F.round(F.max('total_amount'), 2).alias('max_total_amount')) 
    
    df_min_amount = df_min_max_amount \
        .select(F.col('date'), F.col('passenger_count_new'), F.col('min_total_amount')) \
        .groupBy('date').pivot('passenger_count_new').min('min_total_amount') \
        .withColumnRenamed('0', 'min_total_amount_zero') \
        .withColumnRenamed('1', 'min_total_amount_1p') \
        .withColumnRenamed('2', 'min_total_amount_2p') \
        .withColumnRenamed('3', 'min_total_amount_3p') \
        .withColumnRenamed('4', 'min_total_amount_4p') \
        .withColumnRenamed('5', 'min_total_amount_4p_plus') 
        
    
    df_max_amount = df_min_max_amount \
        .select(F.col('date'), F.col('passenger_count_new'), F.col('max_total_amount')) \
        .groupBy('date').pivot('passenger_count_new').max('max_total_amount') \
        .withColumnRenamed('0', 'max_total_amount_zero') \
        .withColumnRenamed('1', 'max_total_amount_1p') \
        .withColumnRenamed('2', 'max_total_amount_2p') \
        .withColumnRenamed('3', 'max_total_amount_3p') \
        .withColumnRenamed('4', 'max_total_amount_4p') \
        .withColumnRenamed('5', 'max_total_amount_4p_plus')  
    
    # Расчет количества поездок для каждого календарного дня (днем считается дата начала поездки)
    df_count_daily = df_taxi.groupby('date').agg(F.count('*').alias('count_daily'))
    # Определение доли поездок с каждой группой количества пассажиров
    df_result = df_taxi \
        .groupby('date', 'passenger_count_new').agg(F.count('*').alias('count_by_pass')) \
        .join(df_count_daily, ['date'], how='inner') \
        .withColumn('percentage', F.round(F.col('count_by_pass') / F.col('count_daily')*100, 2)) \
        .select(F.col('date'), F.col('passenger_count_new'), F.col('percentage')) \
        .groupBy('date').pivot('passenger_count_new').sum('percentage') \
        .fillna(0.0) \
        .withColumnRenamed('0', 'percentage_zero') \
        .withColumnRenamed('1', 'percentage_1p') \
        .withColumnRenamed('2', 'percentage_2p') \
        .withColumnRenamed('3', 'percentage_3p') \
        .withColumnRenamed('4', 'percentage_4p') \
        .withColumnRenamed('5', 'percentage_4p_plus') \
        .join(df_min_amount, ['date'], how='left') \
        .join(df_max_amount, ['date'], how='left') \
        .sort(F.col('date')) 
    
    # Сохранение в формате parquet
    #pd_result = df_result.toPandas()
    #pd_result.to_parquet(path_write)

    # TASK 2 провести аналитику «Как пройденное расстояние и количество пассажиров влияет на чаевые 
    # Векторизуем признаки
    vectorAssembler = VectorAssembler(inputCols = ['trip_distance', 'passenger_count'],
                                       outputCol = 'features')
    vector_df = vectorAssembler.transform(df_taxi)
    vector_df = vector_df.select(['features', 'tip_amount'])
    # Разделяем датасет на тренировочный и тестовый
    splits = vector_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    # Строим модель регрессии
    lr = LinearRegression(featuresCol = 'features', labelCol='tip_amount', maxIter=100, regParam=0., elasticNetParam=0.)
    lr_model = lr.fit(train_df)
    trainingSummary = lr_model.summary
    # Тестируем модель на отложенной выборке
    lr_predictions = lr_model.transform(test_df)
    lr_evaluator = RegressionEvaluator(predictionCol='prediction', \
                                       labelCol='tip_amount', metricName='r2')

    # Сэмплируем данные для графика (чтобы не рисовать по всем данным)
    df_sample = df_taxi \
        .select(F.col('trip_distance'), F.col('passenger_count'), F.col('tip_amount')) \
        .sample(0.001) 
    
    df_sample.toPandas().to_csv(path_for_sample, index = False)
    
    # Выводим результат регрессии
    print("RMSE for train dataset: %f" % trainingSummary.rootMeanSquaredError)
    print("r^2 on train dataset: %f" % trainingSummary.r2)
    print("r^2 on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    # Вывести статистику по числовым столбцам
    #df_taxi.describe().show()
    #df_taxi.printSchema()
    #count_nulls(spark, df_taxi)
    #columns = ['date', 'VendorID', 'passenger_count', 'RatecodeID','store_and_fwd_flag', 'payment_type', 'passenger_count_new']
    #unique_values_dict = unique_values_in_columns(spark, df_taxi, columns)
    #print(unique_values_dict)

    print('################### END ##########################')


def count_nulls(spark, df):
    # Общее количество строк в датафрейме
    total_rows = df.count()
    # Создаем выражения для подсчета пустых ячеек в каждом столбце
    exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c + '_null_count') for c in df.columns]
    # Используем agg для объединения результатов подсчета
    null_counts = df.agg(*exprs)
    # Выводим результаты
    null_counts.show(truncate=False)
    print(f"Общее количество строк: {total_rows}")


def unique_values_in_columns(spark, df, columns):
    unique_values_dict = {} 
    for column in columns:
        unique_values = [row[column] for row in df.select(column).distinct().collect()]
        unique_values_dict[column] = unique_values
    
    return unique_values_dict


if __name__ == '__main__':
    main()
