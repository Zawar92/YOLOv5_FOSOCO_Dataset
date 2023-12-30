# Moving Large data.

source_folder="./prepare_clean_data/augmentation/labels/"
destination_folder="./yolov5/data/train/labels/"

# Create the destination folder if it doesn't exist
# mkdir -p "$destination_folder"

# Move files one by one
for file in "$source_folder"/*; do
    mv "$file" "$destination_folder"
    echo "Moved: $file"
done


####
#### Moving data with requirements.
####

# sourceFolder="./yolov5/data/valid/labels"
# destinationFolder="./prepare_clean_data/valid_robo/labels"

# for file in "$sourceFolder"/*; do
#     validFile=false

#     case "$(basename "$file")" in
#         *_png*|*_jpg*|*_blue*|*_yellow*|*_large_orange*|*_orange*|*_unknown*)
#             validFile=true
#             ;;
#     esac

#     if [ "$validFile" != true ]; then
#         mv "$file" "$destinationFolder"
#         echo "Moved: $file"
#     fi

#     case "$(basename "$file")" in
#         *rf*)
#             mv "$file" "$destinationFolder"
#             echo "Moved: $file"
#             ;;
#     esac
# done