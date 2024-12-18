# Set AWS Region
Set-DefaultAWSRegion -Region "eu-west-2"

# Create a new folder in S3
Write-S3Object -BucketName "bmf-analytics-dev-eu-west-2-leobrix" -Key "newfolder/" -ContentBody ""

# Upload local files and subfolders to the new S3 folder
Write-S3Object -BucketName "bmf-analytics-dev-eu-west-2-leobrix" -KeyPrefix "newfolder/" -Folder "C:\LocalFolder" -Recurse

# Verify upload
Get-S3Object -BucketName "bmf-analytics-dev-eu-west-2-leobrix" -KeyPrefix "newfolder/"
