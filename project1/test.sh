#!/bin/sh

#Root JPG
for file in res/*.jpg
do
    if [ "$file" != "res/*.jpg" ] 
    then
        ( python3 src/project.py $file & );
    fi 
done

#Root PNG
for file in res/*.png
do
    if [ "$file" != "res/*.png" ] 
    then
        ( python3 src/project.py $file & );
    fi 
done

#Signs JPG
for file in res/signs/*.jpg
do
    if [ "$file" != "res/signs/*.jpg" ] 
    then
        ( python3 src/project.py $file & );
    fi 
done

#Signs PNG
for file in res/signs/*.png
do
    if [ "$file" != "res/signs/*.png" ] 
    then
        ( python3 src/project.py $file & );
    fi 
done