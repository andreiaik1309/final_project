# Final_project

Это проект с итоговой работой в рамках обучения Data Engineer 

## Project structure

final_work.py скрипт с кодом для формирования витрины данных;
result папка с файлом (res_table.parquet)  итогового результата работы скрипта 

## Installation

Шаг 1: скачайте скрипт на один из серверов вашего кластера

Шаг 2: положите данные (исходный датасет) в формате csv в папку data в том директории, где будет находится скрипт

Шаг 3: Запуск скрипта:
spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python --conf spark.yarn.dist.archives=hdfs:///user/aloha/spark/share/arima_env.tar.gz#environment --master yarn --conf spark.executor.instances=40 --conf spark.executor.memory=2G  --deploy-mode client /home/prophet/conda/yakovlev/1t_data/final_work.py

примечание: 
в команде запуска скрипта /home/prophet/conda/yakovlev/1t_data - замените на ваш путь
в скрипте измените пути:
path_read = 'file:///home/prophet/conda/yakovlev/1t_data/data/yellow_tripdata_2020-01.csv'
path_write = '/home/prophet/conda/yakovlev/1t_data/result/res_table.parquet'
 path_for_sample = '/home/prophet/conda/yakovlev/1t_data/result/sample_df.csv'


## Result

Ознакомиться с результатами проекта в “итоговая презентация.pdf”

