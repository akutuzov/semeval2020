from docopt import docopt
import logging
import time


def main():
    """
    Assign values in a tab-separated CSV file to two classes according to specified threshold.
    """

    # Get the arguments
    args = docopt("""Assign values in a tab-separated CSV file to two classes according to specified threshold.

    Usage:
        class.py <valueFile> <outPath> <threshold>
        
    Arguments:
        <valueFile> = strings in first column and values in second column
        <outPath> = output path for result file
        <threshold> = values < threshold below threshold are assigned to class 0, >= threshold to class 1

    Note:
        Assumes tap-separated CSV file as input. Appends nan if value in valueFile is nan.

    """)
    
    valueFile = args['<valueFile>']
    outPath = args['<outPath>']
    threshold = float(args['<threshold>'])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()    
    
    # Get string-value tuples
    with open(valueFile, 'r', encoding='utf-8') as f_in:
        string_value = [( line.strip().split('\t')[0], float(line.strip().split('\t')[1]) ) for line in f_in]
        
    # Get strings and string-value map
    strings = [s for (s,v) in string_value]
    string2value = dict(string_value)

    # Print classification to output file
    with open(outPath, 'w', encoding='utf-8') as f_out:
        for string in strings:
            if string2value[string]<threshold:
                f_out.write('\t'.join((string, str(0)+'\n')))
            elif string2value[string]>=threshold:
                f_out.write('\t'.join((string, str(1)+'\n')))
            else:
                f_out.write('\t'.join((string, 'nan'+'\n')))
    

    logging.info("--- %s seconds ---" % (time.time() - start_time))                   

    
if __name__ == '__main__':
    main()
