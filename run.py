import sys
import os
import getopt
import shutil

import configparser

import time

import policy.zeros

# Initiate Training

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print(f'Input file is , {inputfile}')
   print(f'Output file is , {outputfile}')

if __name__ == "__main__":

    LOG_PATH='./logs'
    MODEL_PATH='./model'
    RENDER_PATH='./render'

    if os.path.exists(LOG_PATH):
        shutil.rmtree('./logs', ignore_errors=False)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree('./model', ignore_errors=False)
    if os.path.exists(RENDER_PATH):
        shutil.rmtree('./render', ignore_errors=False)
    os.makedirs(LOG_PATH)
    os.makedirs(MODEL_PATH)
    os.makedirs(RENDER_PATH)

    main(sys.argv[1:])

