# Set the remote path
remote_path='enseirbmatmeca@calculus:/mnt/data2/enseirbmatmeca-data/drawing_manipulation/'

# Set the name of the .gitignore file
gitignore_file='.gitignore'

# Get the list of files and directories to ignore from the .gitignore file
ignore_list=()
while IFS= read -r line; do
    ignore_list+=(--exclude="$line")
done < "$gitignore_file"

ignore_list+=(--exclude=".git")

# Use the rsync command to copy the current directory to the remote path, ignoring the files and directories specified in the .gitignore file
rsync -avz -e 'ssh -i ~/.ssh/enseirb-matmeca.pem' "${ignore_list[@]}" . "$remote_path"