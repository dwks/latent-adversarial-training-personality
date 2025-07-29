#!/bin/sh
# for linh

# modes: all, train, basic_test, eval

time python lat_personality.py --cache-dir cache --data-folder dataset/D5/ --project-name LAT-D5-adam --mode all --num-steps 10
