#!/bin/bash

# Loop over each year
for year in {2018..2023}; do
    # Loop over each month
    for month in {05..12}; do
        # Form the URL
        url="https://database.lichess.org/standard/lichess_db_standard_rated_${year}-${month}.pgn.zst"

        # Check if the URL exists
        if wget --spider $url 2>/dev/null; then
            # Download the file
            wget "$url"

            # The downloaded file name
            filename="lichess_db_standard_rated_${year}-${month}.pgn.zst"

        else
            echo "File for ${year}-${month} does not exist."
        fi

        # Stop the loop if we've reached October 2023
        if [ "$year" -eq 2023 ] && [ "$month" = "10" ]; then
            break
        fi
    done

done
