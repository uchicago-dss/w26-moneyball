def clean_all_files():
    import csv
    import os
    import glob

    COLUMNS_TO_KEEP = [
        "season", "country", "matches_played", "wins", "draws", "losses",
        "points_per_game", "performance_rank", "goals_scored", "goals_conceded",
        "goal_difference", "total_goal_count", "minutes_per_goal_scored",
        "minutes_per_goal_conceded", "clean_sheets", "corners_total",
        "average_possession", "shots", "shots_on_target",
        "goals_scored_per_match", "goals_conceded_per_match",
    ]

    files = glob.glob("/Users/jcrruiz/Downloads/international-*.csv")

    for INPUT_CSV in files:
        print("Processing:", INPUT_CSV)

        filename = os.path.basename(INPUT_CSV)
        OUTPUT_CSV = f"/Users/jcrruiz/Downloads/CLEAN_{filename}"

        with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
             open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=COLUMNS_TO_KEEP)

            writer.writeheader()
            for row in reader:
                filtered_row = {k: row[k] for k in COLUMNS_TO_KEEP if k in row}
                writer.writerow(filtered_row)

    print("All files cleaned.")