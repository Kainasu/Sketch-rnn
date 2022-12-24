# Set the remote and local paths
remote_path='enseirbmatmeca@calculus:/mnt/data2/enseirbmatmeca-data/drawing_manipulation/runs/'
local_path='./runs/'

# Use the rsync command to copy the remote folder to the local path
rsync -avz -e 'ssh -i ~/.ssh/enseirb-matmeca.pem' "$remote_path" "$local_path"