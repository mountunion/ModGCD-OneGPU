#!/bin/bash

#  Script for running timing tests on testmodgcd. Initial run will store the test data in a
#  folder named tests.

for (( i = 16384 ; i <= 360448 ; i += 16384 ))
do
  ./testmodgcd $i
done
