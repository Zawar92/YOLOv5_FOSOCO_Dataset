####
####  Move and rename background images.
####

# Path to the source folder
sourceFolder="./background_images/background_images"

# Path to the destination folder
destinationFolder="./background_images_rename_valid"

# Number of items to move
numberOfItems=500

# Move and rename the files
find "$sourceFolder" -type f | head -n "$numberOfItems" | while read file; do
    newFileName="background_$(basename "$file")"
    newFilePath="$destinationFolder/$newFileName"
    mv "$file" "$newFilePath"
    echo "Moved: $newFilePath"
done
