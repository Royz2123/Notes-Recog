#!/bin/bash

echo
prog1="./hough -s $1 -t $2"
prog2="python stage2.py"

printf "Starting Notes Recognition.\n\n"

printf "Compiling Hough Division.\n"
make
printf "Finished Compiling Hough Division.\n\n"

printf "Running Hough Program.\n"
eval $prog1
printf "Finished Running Hough Program.\n\n"

printf "Running Recognition Program.\n"
eval $prog2
printf "Finished Running Recognition Program.\n\n"

printf "Finished Process, output in sound.wav file.\n\n"
