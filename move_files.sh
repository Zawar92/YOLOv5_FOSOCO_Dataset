source_folder="./prepare_clean_data/data/labels/"
destination_folder="./prepare_clean_data/data/train/labels/"

# Create the destination folder if it doesn't exist
# mkdir -p "$destination_folder"

# Move files one by one
for file in "$source_folder"/*; do
    mv "$file" "$destination_folder"
    echo "Moved: $file"
done