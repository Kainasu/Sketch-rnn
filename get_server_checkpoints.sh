# Set the remote and local paths
suffix=$1
remote_path='enseirbmatmeca@calculus:/mnt/data2/enseirbmatmeca-data/drawing_manipulation/checkpoints'$suffix'/'
local_path='./checkpoints'$suffix'/'

# Use the rsync command to copy the remote folder to the local path
rsync -avz -e 'ssh -i ~/.ssh/enseirb-matmeca.pem' "$remote_path" "$local_path"
