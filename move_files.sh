# Moving Large data.

# sourceFolder="./yolov5/data/train/images"
# destinationFolder="./complete_data/images/"

# # Create the destination folder if it doesn't exist
# mkdir -p "$destination_folder"

# # Move files one by one
# for file in "$source_folder"/*; do
#     mv "$file" "$destination_folder"
#     echo "Moved: $file"
# done


# ####
# #### Moving data with requirements.
# ####

sourceFolder="./yolov5/data/train/images"
destinationFolder="./complete_data/images"

for file in "$sourceFolder"/*; do
    validFile=false

    case "$(basename "$file")" in
        *_png*|*_jpg*|*_blue*|*_yellow*|*_large_orange*|*_orange*|*_unknown*)
            validFile=true
            ;;
    esac

    if [ "$validFile" != true ]; then
        mv "$file" "$destinationFolder"
        echo "Moved: $file"
    fi

    case "$(basename "$file")" in
        *rf*)
            mv "$file" "$destinationFolder"
            echo "Moved: $file"
            ;;
    esac
done

###
### MOVE CROPS ONLY
###
# sourceFolder="./datasets/data/valid/labels"
# destinationFolder="./prepare_clean_data/data/labels"

# for file in "$sourceFolder"/*; do
#     validFile=false

#     case "$(basename "$file")" in
#         *_blue*|*_yellow*|*_large_orange*|*_orange*|*_unknown*)
#             validFile=true
#             ;;
#     esac

#     if [ "$validFile" != false ]; then
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

###
### MOVE BACKGROUND IMAGES
###
# sourceFolder="./prepare_clean_data/orignal_images/images"
# destinationFolder="./prepare_clean_data/background/images"

# # Enable case-insensitive matching
# shopt -s nocaseglob

# for file in "$sourceFolder"/*; do
#     if [[ "$(basename "$file")" =~ background ]]; then
#         mv "$file" "$destinationFolder"
#         echo "Moved: $file"
#     fi
# done

# # Disable case-insensitive matching (optional)
# shopt -u nocaseglob