



# Developed in Python 3.6.7

# Code for running join of a large keyword-csv-file with a smaller category-csv-file from command line.

import argparse
import pandas as pd
import dask.dataframe as dd


def main_merge(args):
    # Read keyword-csv-file as dast dataframe (sections in RAM)
    kw_filename = args.path_kw
    print(f"Loading keywords from: {kw_filename}")
    kw_data = dd.read_csv(kw_filename, sep=args.kw_delimiter)

    # Read category-csv-file as panda dataframe (all in RAM)
    cat_filename = args.path_cat
    print(f"Loading categories from: {cat_filename}")
    cat_data = pd.read_csv(cat_filename, sep=args.cat_delimiter)

    # Make merged data
    merged_data = kw_data.merge(cat_data, how='left', left_on=args.kw_column, right_on=args.cat_column)

    # Write merged data to separate file
    out_filename = args.path_out
    print(f"Writing data to: {out_filename}")
    merged_data.to_csv(out_filename, single_file=True, index=False, sep=args.out_delimiter)

    print("DONE!")


if __name__ == '__main__':
    # parse command line arguments
    argparser = argparse.ArgumentParser(description='Running join of a large keyword-csv-file with a smaller category-csv-file')

    argparser.add_argument('path_kw', type=str, help='Path to the keyword csv file.')
    argparser.add_argument('path_cat', type=str, help='Path to the category csv file.')
    argparser.add_argument('path_out', type=str, help='Path to the output csv file.')
    argparser.add_argument('--kw_delimiter', '-kd', type=str, default=',', help='Delimiter used in the keywords csv file. (default: \',\')')
    argparser.add_argument('--cat_delimiter', '-cd', type=str, default=',', help='Delimiter used in the categories csv file. (default: \',\')')
    argparser.add_argument('--out_delimiter', '-od', type=str, default=',', help='Delimiter used in the output csv file. (default: \',\')')
    argparser.add_argument('--kw_column', '-kc', type=str, default='keyword', help='Name of column containing keywords in the keywords csv file. (default: \'keyword\')')
    argparser.add_argument('--cat_column', '-cc', type=str, default='keyword', help='Name of column containing keywords in the category csv file. (default: \'keyword\')')

    args = argparser.parse_args()

    main_merge(args)


