#!/bin/bash

# First pull to initiate the merge
git pull --no-rebase || true  # The "|| true" ensures the script continues even if git pull exits with error due to conflicts

# Files to take from remote
remote_files=(
  "cowstudyapp/analysis/HMM/run_hmm.r"
  "cowstudyapp/run_analysis.py"
  "cowstudyapp/visuals/make_heatmap_of_predictions.py"
  "cowstudyapp/visuals/show_effects_glmm.py"
  "cowstudyapp/visuals/show_temp_vs_grazing_new.py"
)

# Keep local version for all other files that have conflicts
git checkout --ours .

# Take remote version for specified files
for file in "${remote_files[@]}"; do
  git checkout --theirs "$file"
  echo "Taking remote version of: $file"
done

# Mark all conflicts as resolved
git add .

# Prompt before committing
echo "All conflicts have been resolved."
echo "Review the changes if needed, then complete the merge with 'git commit'."
echo "To commit now, press Enter. To exit without committing, press Ctrl+C."
read -p ""

# Complete the merge
git commit -sm "Merge remote changes, keeping specific files from origin"