#!/bin/sh

for file in res/*.jpg
do
    if [ "$file" != "res/*.jpg" ] 
    then
        ( python3 src/project.py $file & );
    fi 
done