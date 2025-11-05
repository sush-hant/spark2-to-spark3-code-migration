#!/usr/bin/env python3
"""
Spark2 vs Spark3 Output Validation Tool

This tool compares tables between two different Spark pipeline versions (e.g., Spark 2 vs Spark 3)
to validate that outputs match within specified tolerances.

Usage:
    python spark_comparison_tool.py --tables table1 table2 --format parquet \
        --control-root s3://bucket/spark2/output --target-root s3://bucket/spark3/output

    python spark_comparison_tool.py \
        --control-tables catalog.db.table1@snapshot1 catalog.db.table2@snapshot2 \
        --target-tables catalog.db.table1@snapshot3 catalog.db.table2@snapshot4
"""

import argparse
import sys
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import FractionalType


if sys.version_info < (3, 7):
    sys.exit("Error: This script requires Python 3.7 or higher")


def eprint(*args, **kwargs):
    """Print to stderr for logging/debugging purposes."""
    print(*args, file=sys.stderr, **kwargs)


def error(msg: str):
    """Print error message and exit with non-zero status."""
    eprint(f"ERROR: {msg}")
    sys.exit(1)


def extract_catalog(table_name: str) -> str:
    """
    Extract the catalog name from a fully qualified table name.
    
    Args:
        table_name: Table name in format 'catalog.schema.table' or 'schema.table'
    
    Returns:
        Catalog name, defaults to 'spark_catalog' if not specified
    """
    if "." in table_name:
        return table_name.split(".")[0]
    else:
        return "spark_catalog"


def get_ancestors(spark: SparkSession, table_name: str, snapshot: str) -> List[Row]:
    """
    Get the ancestors of a given Iceberg table at a given snapshot.
    
    Args:
        spark: Active SparkSession
        table_name: Fully qualified table name
        snapshot: Snapshot ID to query
    
    Returns:
        List of ancestor snapshot rows
    """
    catalog_name = extract_catalog(table_name)
    return spark.sql(
        f"""CALL {catalog_name}.system.ancestors_of(
        snapshot_id => {snapshot}, table => '{table_name}')"""
    ).collect()


def create_changelog_view(
    spark: SparkSession,
    table_name: str,
    start_snapshot: str,
    end_snapshot: str,
    view_name: str
) -> DataFrame:
    """
    Create an Iceberg changelog view for the provided table between two snapshots.
    
    Args:
        spark: Active SparkSession
        table_name: Fully qualified table name
        start_snapshot: Starting snapshot ID
        end_snapshot: Ending snapshot ID
        view_name: Name for the changelog view
    
    Returns:
        DataFrame representing the changelog view
    """
    catalog_name = extract_catalog(table_name)
    return spark.sql(
        f"""CALL {catalog_name}.system.create_changelog_view(
        table => '{table_name}',
        options => map('start-snapshot-id','{start_snapshot}','end-snapshot-id', '{end_snapshot}'),
        changelog_view => '{view_name}'
        )"""
    )


def drop_iceberg_internal_columns(df: DataFrame) -> DataFrame:
    """
    Drop Iceberg internal columns from a changelog view that would make comparisons tricky.
    
    We keep '_change_type' because if one version inserts and the other deletes,
    that's a difference we want to catch. However, '_change_ordinal' and 
    '_commit_snapshot_id' are expected to differ even with identical end table states.
    
    Args:
        df: DataFrame with Iceberg internal columns
    
    Returns:
        DataFrame with internal columns removed
    """
    new_df = df
    internal = {"_change_ordinal", "_commit_snapshot_id"}
    for c in df.columns:
        name = c.split("#")[0]
        if name in internal:
            new_df = new_df.drop(c)
    return new_df


def get_cdc_views(
    spark: SparkSession,
    ctrl_name: str,
    target_name: str
) -> Tuple[DataFrame, DataFrame]:
    """
    Get the changelog/CDC views of two Iceberg tables with a common ancestor.
    
    This is an optimization for comparing large tables by only comparing the changes
    since a common ancestor snapshot rather than the entire tables.
    
    Args:
        spark: Active SparkSession
        ctrl_name: Control table name with snapshot (format: 'table@snapshot_id')
        target_name: Target table name with snapshot (format: 'table@snapshot_id')
    
    Returns:
        Tuple of (control_changelog_view, target_changelog_view)
    
    Raises:
        Exception if tables don't share a common ancestor or CDC views cannot be created
    """
    (ctrl_table, c_snapshot) = ctrl_name.split("@")
    (target_table, t_snapshot) = target_name.split("@")
    
    if ctrl_table != target_table:
        error(f"{ctrl_table} and {target_table} are not the same table.")
    
    ancestors_c = get_ancestors(spark, ctrl_table, c_snapshot)
    ancestors_t = get_ancestors(spark, target_table, t_snapshot)
    control_ancestor_set = set(ancestors_c)
    shared_ancestor = None
    
    for t in reversed(ancestors_t):
        if t in control_ancestor_set:
            shared_ancestor = t
            break
    
    if shared_ancestor is None:
        error(f"No shared ancestor between tables c:{ancestors_c} t:{ancestors_t}")
    
    try:
        create_changelog_view(
            spark, ctrl_table, shared_ancestor.snapshot_id, c_snapshot, "c"
        )
        create_changelog_view(
            spark, target_table, shared_ancestor.snapshot_id, t_snapshot, "t"
        )
        c_diff_view = drop_iceberg_internal_columns(spark.sql("SELECT * FROM c"))
        t_diff_view = drop_iceberg_internal_columns(spark.sql("SELECT * FROM t"))
    except Exception as e:
        error(f"Iceberg may not support changelog view, doing legacy compare: {e}")
    
    return (c_diff_view, t_diff_view)


def compare_tables(
    spark: SparkSession,
    control: DataFrame,
    target: DataFrame,
    args: argparse.Namespace,
    table_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Compare two DataFrames and report differences.
    
    This function performs a comprehensive comparison including:
    - Schema validation
    - Row count comparison
    - Data difference detection (with optional precision rounding for floats)
    - Duplicate handling
    
    Args:
        spark: Active SparkSession
        control: Control/baseline DataFrame (e.g., Spark 2 output)
        target: Target DataFrame to compare (e.g., Spark 3 output)
        args: Parsed command-line arguments containing comparison settings
    
    Raises:
        Exception if schemas don't match or differences exceed tolerance
    """
    if control.schema != target.schema:
        eprint("Schema mismatch detected:")
        eprint("Control schema:")
        control.printSchema()
        eprint("Target schema:")
        target.printSchema()
        raise Exception("Control schema and target schema do not match")
    
    if args.compare_precision is not None:
        eprint(f"Applying precision rounding to {args.compare_precision} decimal places")
        schema = control.schema
        for c in control.columns:
            if isinstance(schema[c].dataType, FractionalType):
                control = control.withColumn(
                    c, control[c].cast('double').cast(f'decimal(38,{args.compare_precision})')
                )
                target = target.withColumn(
                    c, target[c].cast('double').cast(f'decimal(38,{args.compare_precision})')
                )
    
    control.persist()
    target.persist()
    control_count = control.count()
    target_count = target.count()
    
    eprint(f"Control count: {control_count}, Target count: {target_count}")
    
    try:
        missing_rows = control.subtract(target)
        new_rows = target.subtract(control)
    except Exception as e:
        eprint(f"Warning: subtract() failed, converting all columns to strings: {e}")
        columns = control.columns
        for c in columns:
            control = control.withColumn(c, control[c].cast('string'))
            target = target.withColumn(c, target[c].cast('string'))
        missing_rows = control.subtract(target)
        new_rows = target.subtract(control)
    
    new_rows.cache()
    missing_rows.cache()
    new_rows_count = new_rows.count()
    
    if new_rows_count > 0:
        eprint(f"Found {new_rows_count} rows in target that were not in control")
        new_rows.show(truncate=False)
    
    missing_rows_count = missing_rows.count()
    if missing_rows_count > 0:
        eprint(f"Found {missing_rows_count} rows missing from target (present in control)")
        missing_rows.show(truncate=False)
    
    changed_rows = new_rows_count + missing_rows_count
    row_diff_tol = args.row_diff_tolerance
    exact_tol = row_diff_tol * control_count
    
    passed = changed_rows <= exact_tol
    
    if not passed:
        eprint(
            f"Data differs by {changed_rows} rows, exceeding tolerance of "
            f"{100 * row_diff_tol}% ({exact_tol} rows)."
        )
    else:
        eprint(
            f"Different rows: {changed_rows}, within tolerance of "
            f"{row_diff_tol}% ({exact_tol} rows)"
        )
    
    result = {
        "table_name": table_name,
        "control_count": control_count,
        "target_count": target_count,
        "missing_rows_count": missing_rows_count,
        "new_rows_count": new_rows_count,
        "total_differences": changed_rows,
        "tolerance": row_diff_tol,
        "tolerance_threshold": exact_tol,
        "passed": passed,
        "timestamp": datetime.now().isoformat()
    }
    
    if hasattr(args, 'output_path') and args.output_path:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{args.output_path.rstrip('/')}/{table_name}_{timestamp_str}"
        
        if missing_rows_count > 0:
            eprint(f"Saving missing rows to {output_base}_missing_rows.csv")
            try:
                missing_rows.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}_missing_rows.csv")
            except Exception as e:
                eprint(f"Warning: Could not save missing rows: {e}")
        
        if new_rows_count > 0:
            eprint(f"Saving new rows to {output_base}_new_rows.csv")
            try:
                new_rows.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}_new_rows.csv")
            except Exception as e:
                eprint(f"Warning: Could not save new rows: {e}")
    
    if control_count != target_count:
        eprint(f"Row counts do not match! Control: {control_count}, Target: {target_count}")
        try:
            eprint("Attempting grouped comparison to handle potential duplicates...")
            counted_control = control.groupBy(*control.columns).count().persist()
            control_count = counted_control.count()
            counted_target = target.groupBy(*target.columns).count().persist()
            new_rows = counted_target.subtract(counted_control)
            missing_rows = counted_control.subtract(counted_target)
            new_rows_count = new_rows.count()
            
            if new_rows_count > 0:
                eprint(f"Found {new_rows_count} grouped rows that were not in control")
                new_rows.show(truncate=False)
            
            missing_rows_count = missing_rows.count()
            if missing_rows_count > 0:
                eprint(f"Found {missing_rows_count} grouped rows missing from target")
                missing_rows.show(truncate=False)
            
            exact_tol = row_diff_tol * control_count
            changed_rows = new_rows_count + missing_rows_count
            
            result["grouped_comparison"] = True
            result["grouped_differences"] = changed_rows
            
            if changed_rows > exact_tol:
                result["passed"] = False
                eprint(
                    f"Grouped data differs by {changed_rows} rows, exceeding tolerance of "
                    f"{100 * row_diff_tol}% ({exact_tol} rows)."
                )
            else:
                eprint(
                    f"Grouped different rows: {changed_rows}, within tolerance of "
                    f"{row_diff_tol}% ({exact_tol} rows)"
                )
        except Exception as e:
            result["grouped_comparison_error"] = str(e)
            result["passed"] = False
            eprint(f"Data counts differ and grouped comparison failed: {e}")
    
    if not result["passed"]:
        error(f"Comparison failed for table {table_name}")
    
    return result


def run_comparisons(spark: SparkSession, tables: List[Tuple[str, str]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run comparisons for multiple table pairs.
    
    Args:
        spark: Active SparkSession
        tables: List of (control_table, target_table) tuples
        args: Parsed command-line arguments
    
    Returns:
        List of comparison results for each table
    """
    results = []
    for (ctrl_name, target_name) in tables:
        eprint(f"\n{'='*80}")
        eprint(f"Comparing: {ctrl_name} vs {target_name}")
        eprint(f"{'='*80}")
        
        table_name = ctrl_name.split("@")[0] if "@" in ctrl_name else ctrl_name
        
        if "@" in ctrl_name:
            try:
                eprint("Attempting to use CDC views for optimized comparison...")
                (c_diff_view, t_diff_view) = get_cdc_views(spark, ctrl_name, target_name)
                eprint(f"Using CDC views for comparison")
                result = compare_tables(spark, c_diff_view, t_diff_view, args, table_name)
                results.append(result)
            except Exception as e:
                eprint(f"CDC view comparison failed, falling back to full table compare: {e}")
                (ctrl_table, c_snapshot) = ctrl_name.split("@")
                (target_table, t_snapshot) = target_name.split("@")
                control_df = spark.read.option("snapshot-id", c_snapshot).table(ctrl_table)
                target_df = spark.read.option("snapshot-id", t_snapshot).table(target_table)
                result = compare_tables(spark, control_df, target_df, args, table_name)
                results.append(result)
        else:
            control_df = spark.read.table(ctrl_name)
            target_df = spark.read.table(target_name)
            result = compare_tables(spark, control_df, target_df, args, table_name)
            results.append(result)
    
    return results


def main():
    """Main entry point for the Spark comparison tool."""
    parser = argparse.ArgumentParser(
        description='Compare two different versions of a Spark pipeline (e.g., Spark 2 vs Spark 3). '
                    'Either --tables with --control-root and --target-root, or '
                    '--control-tables and --target-tables must be specified.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--tables',
        type=str,
        nargs='+',
        required=False,
        help='Names of the tables to compare (used with --control-root and --target-root)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='parquet',
        help='Format of the table files (default: parquet)'
    )
    parser.add_argument(
        '--control-root',
        type=str,
        help='Root directory for the control/baseline files (e.g., Spark 2 output)'
    )
    parser.add_argument(
        '--target-root',
        type=str,
        help='Root directory for the target files (e.g., Spark 3 output)'
    )
    parser.add_argument(
        '--control-tables',
        type=str,
        nargs='+',
        help='Control table names (can include @snapshot_id for Iceberg tables)'
    )
    parser.add_argument(
        '--target-tables',
        type=str,
        nargs='+',
        help='Target table names (can include @snapshot_id for Iceberg tables)'
    )
    parser.add_argument(
        '--compare-precision',
        type=int,
        help='Number of decimal places for fractional comparisons (e.g., 2 for 0.01 precision)'
    )
    parser.add_argument(
        '--row-diff-tolerance',
        type=float,
        default=0.0,
        help='Tolerance for percentage of different rows (0.0 to 1.0, default: 0.0 for exact match)'
    )
    parser.add_argument(
        '--csv-header',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Whether CSV files have a header row (default: true)'
    )
    parser.add_argument(
        '--csv-infer-schema',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Whether to infer schema for CSV files (default: true)'
    )
    parser.add_argument(
        '--csv-delimiter',
        type=str,
        default=',',
        help='Delimiter for CSV files (default: ,)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        help='S3 path to save comparison results (JSON summary and CSV diffs). Example: s3://bucket/comparison-results'
    )
    
    args = parser.parse_args()
    
    if args.control_root is None and args.control_tables is None:
        parser.error("Either --control-root or --control-tables must be specified")
    
    if args.control_root is not None and args.tables is None:
        parser.error("--tables must be specified when using --control-root")
    
    if args.control_tables is not None and args.target_tables is None:
        parser.error("--target-tables must be specified when using --control-tables")
    
    if args.control_tables is not None and len(args.control_tables) != len(args.target_tables):
        parser.error("Number of control tables must match number of target tables")
    
    spark = SparkSession.builder \
        .appName("Spark2-vs-Spark3-Comparison") \
        .getOrCreate()
    
    try:
        all_results = []
        
        if args.control_root is not None:
            eprint(f"Comparing tables from directories:")
            eprint(f"  Control: {args.control_root}")
            eprint(f"  Target: {args.target_root}")
            eprint(f"  Format: {args.format}")
            
            for table in args.tables:
                eprint(f"\n{'='*80}")
                eprint(f"Comparing table: {table}")
                eprint(f"{'='*80}")
                
                control_path = f"{args.control_root.rstrip('/')}/{table}"
                target_path = f"{args.target_root.rstrip('/')}/{table}"
                
                reader_control = spark.read.format(args.format)
                reader_target = spark.read.format(args.format)
                
                if args.format.lower() == 'csv':
                    reader_control = reader_control \
                        .option("header", args.csv_header) \
                        .option("inferSchema", args.csv_infer_schema) \
                        .option("sep", args.csv_delimiter)
                    reader_target = reader_target \
                        .option("header", args.csv_header) \
                        .option("inferSchema", args.csv_infer_schema) \
                        .option("sep", args.csv_delimiter)
                
                control = reader_control.load(control_path)
                target = reader_target.load(target_path)
                result = compare_tables(spark, control, target, args, table)
                all_results.append(result)
        else:
            eprint(f"Comparing {len(args.control_tables)} table pairs")
            tables = list(zip(args.control_tables, args.target_tables))
            all_results = run_comparisons(spark, tables, args)
        
        if args.output_path:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = f"{args.output_path.rstrip('/')}/comparison_summary_{timestamp_str}.json"
            
            summary = {
                "comparison_timestamp": datetime.now().isoformat(),
                "total_tables": len(all_results),
                "passed_tables": sum(1 for r in all_results if r["passed"]),
                "failed_tables": sum(1 for r in all_results if not r["passed"]),
                "control_root": args.control_root if args.control_root else "N/A",
                "target_root": args.target_root if args.target_root else "N/A",
                "tolerance": args.row_diff_tolerance,
                "results": all_results
            }
            
            eprint(f"\nSaving comparison summary to {summary_path}")
            try:
                summary_json = json.dumps(summary, indent=2)
                summary_rdd = spark.sparkContext.parallelize([summary_json])
                summary_rdd.coalesce(1).saveAsTextFile(summary_path)
                eprint(f"Summary saved successfully")
            except Exception as e:
                eprint(f"Warning: Could not save summary JSON: {e}")
        
        print("\n" + "="*80)
        print("SUCCESS: All table comparisons completed within tolerance")
        print("="*80)
        
        if args.output_path:
            print(f"Results saved to: {args.output_path}")
            print(f"Summary: {summary_path}")
    except Exception as e:
        eprint(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
