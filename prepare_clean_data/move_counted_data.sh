#!/bin/bash

# Source folders
imagesFolder="../yolov5/data/valid/images"
labelsFolder="../yolov5/data/valid/labels"

# Destination folders
destinationImagesFolder="../complete_data/images"
destinationLabelsFolder="../complete_data/labels"

# Static number of files to be moved
numFiles=99115

# Move files with matching names
count=0

for imageFile in "$imagesFolder"/*.{jpg,png}; do
    imageName=$(basename "$imageFile")
    labelFile="$labelsFolder/${imageName%.*}.txt"

    if [ -e "$labelFile" ]; then
        # Debugging output
        echo "Image File: $imageFile"
        echo "Label File: $labelFile"

        # Move the image file to the destination images folder
        mv "$imageFile" "$destinationImagesFolder/"
        echo "Moved Image: $imageName to $destinationImagesFolder"

        # Move the label file to the destination labels folder
        mv "$labelFile" "$destinationLabelsFolder/"
        echo "Moved Label: $(basename "$labelFile") to $destinationLabelsFolder"

        ((count++))

        # Check if the required number of files has been moved
        if [ "$count" -eq "$numFiles" ]; then
            echo "Required number of files moved to $destinationImagesFolder and $destinationLabelsFolder"
            exit 0
        fi
    else
        # Debugging output
        echo "Label File not found for Image: $imageName"
    fi
done

# If the loop completes without moving the required number of files
echo "Error: Unable to move the required number of files. Insufficient matching files found."
exit 1
