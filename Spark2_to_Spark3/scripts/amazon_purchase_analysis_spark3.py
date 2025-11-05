"""
Amazon Purchase Analysis - AWS EMR Spark 3 Version
===================================================

This script is migrated for AWS EMR with Spark 3.x
Reads data from S3, processes with Spark, and writes results back to S3.

Usage:
    spark-submit \
        --deploy-mode cluster \
        s3://your-bucket/scripts/amazon_purchase_analysis_spark3.py \
        s3://your-bucket/data/input/ \
        s3://your-bucket/data/output/

Arguments:
    input_path: S3 path to input data directory (containing CSV files)
    output_path: S3 path for output results

Author: Migrated to Spark 3
Date: 2025-10-22
Spark Version: 3.x (EMR 6.x or 7.x)

Migration Notes:
- Updated for Spark 3.x compatibility
- Added legacy compatibility settings for smooth migration
- All APIs updated to Spark 3 standards
- Adaptive Query Execution (AQE) enabled by default in Spark 3
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for S3 paths"""
    if len(sys.argv) < 3:
        logger.error("Usage: spark-submit script.py s3://amzn-emr-spark/data/input/ s3://amzn-emr-spark/data/output_csv/")
        logger.error("Example: spark-submit script.py s3://bucket/input/ s3://bucket/output/")
        sys.exit(1)
    
    input_path = sys.argv[1].rstrip('/')
    output_path = sys.argv[2].rstrip('/')
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    return input_path, output_path

orders_schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("order_date", StringType(), False),
    StructField("delivery_date", StringType(), True),
    StructField("product_id", StringType(), False),
    StructField("product_type", StringType(), True),
    StructField("category", StringType(), True),
    StructField("quantity", IntegerType(), False),
    StructField("unit_price", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("discount_percent", DoubleType(), True),
    StructField("discount_amount", DoubleType(), True),
    StructField("final_amount", DoubleType(), True),
    StructField("shipping_method", StringType(), True),
    StructField("shipping_cost", DoubleType(), True),
    StructField("tax_amount", DoubleType(), True),
    StructField("grand_total", DoubleType(), False),
    StructField("payment_method", StringType(), True),
    StructField("region", StringType(), True),
    StructField("city", StringType(), True),
    StructField("customer_segment", StringType(), True),
    StructField("order_status", StringType(), False),
    StructField("rating", DoubleType(), True),
    StructField("is_prime_member", BooleanType(), True)
])

customers_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("account_age_days", IntegerType(), True),
    StructField("total_lifetime_orders", IntegerType(), True),
    StructField("preferred_payment", StringType(), True)
])

products_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("brand", StringType(), True),
    StructField("seller_id", StringType(), True),
    StructField("stock_quantity", IntegerType(), True),
    StructField("avg_rating", DoubleType(), True),
    StructField("num_reviews", IntegerType(), True)
])

def main():
    """Main execution function"""
    
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING AMAZON PURCHASE ANALYSIS ON EMR")
    logger.info("=" * 80)
    
    input_path, output_path = parse_arguments()
    
    try:
        logger.info("Initializing Spark Session for EMR with Spark 3 configurations")
        spark = SparkSession.builder \
            .appName("Amazon Purchase Analysis - EMR Spark 3") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.sql.legacy.allowNegativeScaleOfDecimal", "true") \
            .config("spark.sql.storeAssignmentPolicy", "LEGACY") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        logger.info(f"Spark version: {spark.version}")
        logger.info(f"Spark master: {spark.sparkContext.master}")
        logger.info("Spark 3 compatibility settings enabled for smooth migration")
        
    except Exception as e:
        logger.error(f"Failed to initialize Spark: {str(e)}")
        sys.exit(1)
    
    try:
        logger.info("=" * 80)
        logger.info("LOADING DATA FROM S3")
        logger.info("=" * 80)
        
        orders_path = f"{input_path}/amazon_orders.csv"
        customers_path = f"{input_path}/amazon_customers.csv"
        products_path = f"{input_path}/amazon_products.csv"
        
        logger.info(f"Reading orders from: {orders_path}")
        df_orders = spark.read \
            .option("header", "true") \
            .schema(orders_schema) \
            .csv(orders_path) \
            .withColumn("order_date", F.to_date(F.col("order_date")))
        
        logger.info(f"Reading customers from: {customers_path}")
        df_customers = spark.read \
            .option("header", "true") \
            .schema(customers_schema) \
            .csv(customers_path)
        
        logger.info(f"Reading products from: {products_path}")
        df_products = spark.read \
            .option("header", "true") \
            .schema(products_schema) \
            .csv(products_path)
        
        logger.info("Validating and cleaning data")
        df_orders = df_orders.filter(
            (F.col("order_id").isNotNull()) &
            (F.col("customer_id").isNotNull()) &
            (F.col("product_id").isNotNull()) &
            (F.col("order_date").isNotNull()) &
            (F.col("grand_total").isNotNull()) &
            (F.col("grand_total") >= 0) &
            (F.col("quantity").isNotNull()) &
            (F.col("quantity") > 0)
        )
        
        df_orders = df_orders.withColumn(
            "rating",
            F.when(
                (F.col("rating").isNotNull()) & 
                (F.col("rating") >= 1) & 
                (F.col("rating") <= 5),
                F.col("rating")
            ).otherwise(None)
        )
        
        logger.info("Creating filtered DataFrame for delivered orders")
        df_orders_delivered = df_orders.filter(F.col("order_status") == "Delivered").cache()
        
        orders_count = df_orders.count()
        delivered_count = df_orders_delivered.count()
        customers_count = df_customers.count()
        products_count = df_products.count()
        
        logger.info("=" * 80)
        logger.info("DATA LOADED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total Orders: {orders_count}")
        logger.info(f"Delivered Orders: {delivered_count}")
        logger.info(f"Total Customers: {customers_count}")
        logger.info(f"Total Products: {products_count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 1: Multi-Level Revenue Analysis")
        logger.info("=" * 80)
        
        revenue_analysis = df_orders_delivered \
            .groupBy("region", "category", "payment_method") \
            .agg(
                F.sum("grand_total").alias("total_revenue"),
                F.count("order_id").alias("order_count"),
                F.avg("grand_total").alias("avg_order_value"),
                F.sum("quantity").alias("total_quantity"),
                F.avg("discount_percent").alias("avg_discount_pct"),
                F.countDistinct("customer_id").alias("unique_customers")
            ) \
            .withColumn("revenue_per_customer", 
                        F.when(F.col("unique_customers") > 0,
                               F.round(F.col("total_revenue") / F.col("unique_customers"), 2))
                        .otherwise(0))
        
        logger.info(f"Revenue analysis records: {revenue_analysis.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 2: Product Ranking with Window Functions")
        logger.info("=" * 80)
        
        category_window = Window.partitionBy("category").orderBy(F.desc("product_revenue"))
        region_window = Window.partitionBy("region").orderBy(F.desc("product_revenue"))
        overall_window = Window.orderBy(F.desc("product_revenue"))
        
        product_performance = df_orders_delivered \
            .groupBy("product_id", "product_type", "category", "region") \
            .agg(
                F.sum("grand_total").alias("product_revenue"),
                F.sum("quantity").alias("total_sold"),
                F.countDistinct("customer_id").alias("unique_buyers")
            ) \
            .withColumn("rank_in_category", F.row_number().over(category_window)) \
            .withColumn("rank_in_region", F.row_number().over(region_window)) \
            .withColumn("overall_rank", F.row_number().over(overall_window)) \
            .withColumn("category_revenue_pct", 
                        F.round((F.col("product_revenue") / 
                                F.sum("product_revenue").over(Window.partitionBy("category"))) * 100, 2)) \
            .filter(F.col("rank_in_category") <= 5)
        
        logger.info(f"Product performance records: {product_performance.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 3: RFM (Recency, Frequency, Monetary) Analysis")
        logger.info("=" * 80)
        
        max_date = df_orders_delivered.agg(F.max("order_date")).collect()[0][0]
        logger.info(f"Reference date for RFM: {max_date}")
        
        rfm_analysis = df_orders_delivered \
            .groupBy("customer_id") \
            .agg(
                F.datediff(F.lit(max_date), F.max("order_date")).alias("recency"),
                F.count("order_id").alias("frequency"),
                F.sum("grand_total").alias("monetary")
            )
        
        rfm_window = Window.orderBy(F.asc("recency"))
        freq_window = Window.orderBy(F.desc("frequency"))
        mon_window = Window.orderBy(F.desc("monetary"))
        
        rfm_scored = rfm_analysis \
            .withColumn("r_score", F.ntile(5).over(rfm_window)) \
            .withColumn("f_score", F.ntile(5).over(freq_window)) \
            .withColumn("m_score", F.ntile(5).over(mon_window)) \
            .withColumn("rfm_score", 
                        F.concat(F.col("r_score"), F.col("f_score"), F.col("m_score"))) \
            .withColumn("customer_value",
                        F.when((F.col("r_score") >= 4) & (F.col("f_score") >= 4) & (F.col("m_score") >= 4), "Champions")
                        .when((F.col("r_score") >= 3) & (F.col("f_score") >= 3), "Loyal Customers")
                        .when((F.col("r_score") >= 4) & (F.col("f_score") <= 2), "Promising")
                        .when((F.col("r_score") <= 2) & (F.col("f_score") >= 3), "At Risk")
                        .when((F.col("r_score") <= 2) & (F.col("f_score") <= 2), "Lost")
                        .otherwise("Regular"))
        
        logger.info(f"RFM analysis records: {rfm_scored.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 4: 360° Customer View with Multiple Joins")
        logger.info("=" * 80)
        



                # Disambiguate overlapping column names before joining
        df_customers_ren = df_customers

        df_products_ren = df_products.select(
            "product_id",
            F.col("brand").alias("product_brand"),
            F.col("avg_rating").alias("product_avg_rating"),
            F.col("stock_quantity").alias("product_stock")
        )

        # Keep only needed columns from orders to avoid ambiguity
        orders_for_360 = df_orders_delivered.select(
            "order_id", "customer_id", "product_id", "grand_total",
            "discount_percent", "discount_amount", "rating"
        )

        customer_360 = (
            orders_for_360
              .join(df_customers_ren, "customer_id", "inner")
              .join(df_products_ren, "product_id", "inner")
              .join(
                  rfm_scored.select("customer_id", "customer_value", "rfm_score"),
                  "customer_id", "left"
              )
              .groupBy(
                  "customer_id", "age", "gender", 
                  "customer_value", "account_age_days", "total_lifetime_orders", "preferred_payment"
              )
              .agg(
                  F.count("order_id").alias("total_orders"),
                  F.sum("grand_total").alias("lifetime_value"),
                  F.avg("grand_total").alias("avg_order_value"),
                  F.countDistinct("product_brand").alias("brands_purchased"),
                  F.avg("rating").alias("avg_rating_given"),
                  F.sum(F.when(F.col("discount_percent") > 0, 1).otherwise(0)).alias("orders_with_discount"),
                  F.sum("discount_amount").alias("total_discounts_availed")
              )
              .withColumn(
                  "discount_affinity",
                  F.when(F.col("total_orders") > 0,
                         F.round((F.col("orders_with_discount") / F.col("total_orders")) * 100, 2)
                  ).otherwise(0)
              )
              .withColumn(
                  "daily_value",
                  F.when(F.col("account_age_days") > 0,
                         F.round(F.col("lifetime_value") / F.col("account_age_days"), 2)
                  ).otherwise(0)
              )
        )

        logger.info(f"Customer 360 records: {customer_360.count()}")



    
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 5: Time Series Analysis with Moving Averages")
        logger.info("=" * 80)
        
        daily_sales = df_orders_delivered \
            .groupBy("order_date") \
            .agg(
                F.sum("grand_total").alias("daily_revenue"),
                F.count("order_id").alias("daily_orders"),
                F.avg("grand_total").alias("avg_order_value")
            ) \
            .orderBy("order_date")
        
        days_7_window = Window.orderBy("order_date").rowsBetween(-6, 0)
        days_30_window = Window.orderBy("order_date").rowsBetween(-29, 0)
        
        time_series_analysis = daily_sales \
            .withColumn("revenue_7day_ma", F.round(F.avg("daily_revenue").over(days_7_window), 2)) \
            .withColumn("revenue_30day_ma", F.round(F.avg("daily_revenue").over(days_30_window), 2)) \
            .withColumn("orders_7day_ma", F.round(F.avg("daily_orders").over(days_7_window), 2)) \
            .withColumn("month", F.month(F.col("order_date"))) \
            .withColumn("year", F.year(F.col("order_date"))) \
            .withColumn("day_of_week", F.dayofweek(F.col("order_date")))
        
        logger.info(f"Time series records: {time_series_analysis.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 6: Monthly Cohort Retention Analysis")
        logger.info("=" * 80)
        
        first_purchase = df_orders_delivered \
            .groupBy("customer_id") \
            .agg(F.min("order_date").alias("first_purchase_date")) \
            .withColumn("cohort_month", 
                        F.concat(F.year("first_purchase_date"), 
                                F.lit("-"), 
                                F.lpad(F.month("first_purchase_date"), 2, "0")))
        
        cohort_data = df_orders_delivered \
            .join(first_purchase, "customer_id") \
            .withColumn("order_month", 
                        F.concat(F.year("order_date"), 
                                F.lit("-"), 
                                F.lpad(F.month("order_date"), 2, "0"))) \
            .withColumn("months_since_first", 
                        F.round(F.months_between(F.col("order_date"), F.col("first_purchase_date")), 0).cast("int"))
        
        cohort_analysis = cohort_data \
            .groupBy("cohort_month", "months_since_first") \
            .agg(F.countDistinct("customer_id").alias("active_customers")) \
            .orderBy("cohort_month", "months_since_first")
        
        logger.info(f"Cohort analysis records: {cohort_analysis.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION 7: Statistical Analysis by Category and Region")
        logger.info("=" * 80)
        
        statistical_analysis = df_orders_delivered \
            .groupBy("category", "region") \
            .agg(
                F.count("order_id").alias("order_count"),
                F.sum("grand_total").alias("total_revenue"),
                F.mean("grand_total").alias("mean_order_value"),
                F.stddev("grand_total").alias("stddev_order_value"),
                F.min("grand_total").alias("min_order_value"),
                F.expr("percentile_approx(grand_total, 0.25)").alias("q1_order_value"),
                F.expr("percentile_approx(grand_total, 0.5)").alias("median_order_value"),
                F.expr("percentile_approx(grand_total, 0.75)").alias("q3_order_value"),
                F.max("grand_total").alias("max_order_value"),
                F.avg("rating").alias("avg_rating"),
                F.sum(F.when(F.col("is_prime_member") == True, 1).otherwise(0)).alias("prime_orders"),
                F.sum(F.when(F.col("is_prime_member") == False, 1).otherwise(0)).alias("non_prime_orders")
            ) \
            .withColumn("prime_order_pct", 
                        F.when(F.col("order_count") > 0,
                               F.round((F.col("prime_orders") / F.col("order_count")) * 100, 2))
                        .otherwise(0)) \
            .withColumn("cv_order_value", 
                        F.when((F.col("mean_order_value").isNotNull()) & (F.col("mean_order_value") > 0),
                               F.round((F.col("stddev_order_value") / F.col("mean_order_value")) * 100, 2))
                        .otherwise(None))
        
        logger.info(f"Statistical analysis records: {statistical_analysis.count()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("SAVING TRANSFORMATION RESULTS TO S3")
        logger.info("=" * 80)
        
        partition_count = 10
        
        logger.info(f"Saving revenue_analysis to {output_path}/revenue_analysis/")
        revenue_analysis.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/revenue_analysis")
        
        logger.info(f"Saving product_performance to {output_path}/product_performance/")
        product_performance.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/product_performance")
        
        logger.info(f"Saving rfm_analysis to {output_path}/rfm_analysis/")
        rfm_scored.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/rfm_analysis")
        
        logger.info(f"Saving customer_360 to {output_path}/customer_360/")
        customer_360.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/customer_360")
        
        logger.info(f"Saving time_series to {output_path}/time_series/")
        time_series_analysis.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/time_series")
        
        logger.info(f"Saving cohort_analysis to {output_path}/cohort_analysis/")
        cohort_analysis.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/cohort_analysis")
        
        logger.info(f"Saving statistical_analysis to {output_path}/statistical_analysis/")
        statistical_analysis.coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/statistical_analysis")
        
        logger.info("All transformations completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        
        df_orders_delivered.unpersist()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"ANALYSIS COMPLETED SUCCESSFULLY IN {execution_time:.2f} SECONDS")
        logger.info("=" * 80)
        

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)  # keep
        import traceback; traceback.print_exc()                # ensure it goes to stderr
        raise
    
    finally:
        logger.info("Stopping Spark session")
        spark.stop()

if __name__ == "__main__":
    main()
