# Developed in Python 3.6.7

# Code for aggregating unique keywords from multiple csv files.

import argparse
import csv
import functools
print = functools.partial(print, flush=True)

def load_csv_column_as_set(path, column_name, delimiter=','):
    """
    Load the contents of a column in a csv file.

    Args:
        path: Path to the csv file containing the column.
        delimiter: The delimiter used in the csv file.
        column_name: The name of the target column.

    Returns:
        A list of column fields as strings.
    """
    unique_fields = set()
    # go over all the csv rows
    try:
        print(f"Reading from file : {path}")
        with open(path, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # first line - get the column index from the header
                    keyword_header = row
                    # print(f"Analysing header row: {keyword_header}")
                    column_index = keyword_header.index(column_name)
                else:
                    # get the correct field
                    unique_fields.add(row[column_index])
                line_count += 1
    except Exception as e:
        print(f"Exception reading file {e}")  # Return empty set or whats read.

    # Remove empty string keyword from the set
    unique_fields.discard("")
    return unique_fields


def load_csv_column_and_filter_as_set(path, column_name, filter_column_name, filter_start_value, delimiter=','):
    """
    Load the contents of a column in a csv file if it matches filter_accept_value.

    Args:
        path: Path to the csv file containing the column.
        delimiter: The delimiter used in the csv file.
        column_name: The name of the target column.
        filter_column_name: The name of the filter value column.
        filter_start_value: The filter column has to start with this value.

    Returns:
        A list of unique column fields as strings.
    """
    unique_fields = set()
    # go over all the csv rows
    with open(path, encoding="utf-8") as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=delimiter)
        # Replace all NULL elements in the input csv with empty string. This is the case if no keywords are given.
        csv_reader = csv.reader((line.replace('\0','') for line in csv_file), delimiter=delimiter)
        line_count = 0
        valid_line_count = 0
        for row in csv_reader:
            #print(f"{line_count} | row: {row}")
            if line_count == 0:
                # first line - get the column index from the header
                keyword_header = row
                # print(f"Analysing header row: {keyword_header}")
                column_index = keyword_header.index(column_name)
                filter_column_index = keyword_header.index(filter_column_name)
            else:
                # Check if row has accept_value
                if row[filter_column_index].startswith(filter_start_value):
                    # get the value
                    valid_line_count += 1
                    unique_fields.add(row[column_index])
            line_count += 1
    print(f"unique_fields:{len(unique_fields)} | valid_lines:{valid_line_count} | tot_lines:{line_count}")
    return unique_fields


def main_aggregate(args):
    # get aggregated keywords
    aggregate_filename = args.path_aggregate
    unique_keywords = load_csv_column_as_set(
        aggregate_filename,
        "keyword")
    print(f'Loaded {len(unique_keywords)} aggregated_keywords.')

    # get new keywords
    keyword_filename = args.path_keywords
    print(f"Loading keywords from: {keyword_filename}")
    new_keywords = load_csv_column_and_filter_as_set(
        keyword_filename,
        args.keywords_column,
        args.keywords_filter_column,
        args.keywords_filter_starts_with,
        delimiter = args.keywords_delimiter)
    print(f'Loaded {len(new_keywords)} new_keywords.')

    unique_keywords.update(new_keywords)
    print(f"Total {len(unique_keywords)} keywords after merge")

    # write updated aggregated keywords
    aggregate_filename = args.path_aggregate
    print(f"Writing keywords to: {aggregate_filename}")
    with open(aggregate_filename, "w", encoding="utf8") as outfile:
        outwriter = csv.writer(outfile, delimiter=",", quotechar='"')
        # write header
        out_header = ["keyword"]
        outwriter.writerow(out_header)

        # write results row by row
        for keyword in unique_keywords:
            row = [f"{keyword}"]
            outwriter.writerow(row)

    print("DONE!")


if __name__ == '__main__':
    # parse command line arguments
    argparser = argparse.ArgumentParser(description='Tool for agregating keywords from multiple files')

    argparser.add_argument('path_keywords', type=str, help='Path to the input keywords csv file.')
    argparser.add_argument('path_aggregate', type=str, help='Path to the aggregated keywords csv file.')
    argparser.add_argument('--keywords_delimiter', '-kd', type=str, default=',', help='Delimiter used in the keywords csv file. (default: \',\')')
    argparser.add_argument('--keywords_column', '-kc', type=str, default='Keyword', help='Name of column containing keywords in the keywords csv file. (default: \'Keyword\')')
    argparser.add_argument('--keywords_filter_column', '-kfc', type=str, default='Country', help='Name of column containing string for filtering rows in the keywords csv file. (default: \'Country\')')
    argparser.add_argument('--keywords_filter_starts_with', '-kfa', type=str, default='', help='Value to check if filter column starts with. (default: \'\')')
    args = argparser.parse_args()

    main_aggregate(args)
