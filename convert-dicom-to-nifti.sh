#!/bin/bash

# 🚨 Stop on errors
set -e

# 🛑 Check arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <DICOM_DIR> <OUTPUT_NIFTI> [MIN_UNIQUE_VALUES]"
    exit 1
fi

# 🔹 Read input arguments
DICOM_DIR="$1"
OUTPUT_NIFTI="$2"
MIN_UNIQUE_VALUES="${3:-3}"  # Default to 3 if not provided

# 🔹 Create a temporary directory for sorted DICOMs
SORTED_DIR=$(mktemp -d)

# 🔹 Extract positions & instance numbers
echo "Extracting DICOM slice positions..."
sorted_list=$(mktemp)
for file in "$DICOM_DIR"/*.dcm; do 
    pos=$(dcmdump "$file" | grep "(0020,0032)" | awk -F'[][]' '{print $2}' | tr '\\' ' ') 
    instance=$(dcmdump "$file" | grep "(0020,0013)" | awk '{print $NF}') 
    echo "$pos $instance $file"
done > "$sorted_list"

# 🔹 Detect sorting column
echo "Detecting sorting column..."
sort_column=3  # Default to third column
for col in 1 2 3; do
    unique_count=$(awk "{print \$$col}" "$sorted_list" | sort -u | wc -l)
    if [[ "$unique_count" -ge "$MIN_UNIQUE_VALUES" ]]; then
        sort_column="$col"
        break
    fi
done

echo "Sorting by column $sort_column..."

# 🔹 Sort by detected column
sorted_sorted_list=$(mktemp)
sort -k"$sort_column" -n "$sorted_list" > "$sorted_sorted_list"

# 🔹 Rename & update metadata
echo "Renaming and updating metadata..."
counter=1
while read -r line; do
    file=$(echo "$line" | awk '{print $NF}')
    new_file="$SORTED_DIR/$(printf "%04d.dcm" "$counter")"
    cp "$file" "$new_file"
    
    # Update Instance Number
    dcmodify -nb -i "(0020,0013)=$(printf "%d" "$counter")" "$new_file"

    counter=$((counter + 1))
done < "$sorted_sorted_list"

# 🔹 Convert to NIfTI
echo "Converting to NIfTI format..."
dcm2niix -z y -o "$(dirname "$OUTPUT_NIFTI")" -f "$(basename "$OUTPUT_NIFTI" .nii.gz)" -i y -m y "$SORTED_DIR" 

# 🔹 Cleanup
rm -rf "$SORTED_DIR" "$sorted_list" "$sorted_sorted_list"

echo "✅ Conversion complete! Output: ${OUTPUT_NIFTI}"

