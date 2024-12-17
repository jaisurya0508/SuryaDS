# Parameters
$BucketName = "your-bucket-name"               # Replace with your S3 bucket name
$FolderPrefix = "your/existing/folder/"        # S3 folder prefix (e.g., 'data/files/')
$LocalFolderPath = "C:\path\to\local\files"    # Local folder path containing files to upload

# List existing files in the S3 folder
$existingS3Files = (Get-S3Object -BucketName $BucketName -Prefix $FolderPrefix).Key

# Get all local files in the folder
$localFiles = Get-ChildItem -Path $LocalFolderPath -File

# Upload only new files
foreach ($file in $localFiles) {
    # Define the S3 key (folder prefix + file name)
    $s3Key = "$FolderPrefix$($file.Name)"

    # Check if the file already exists in S3
    if ($existingS3Files -notcontains $s3Key) {
        Write-Output "Uploading $($file.Name) to S3://$BucketName/$s3Key"
        Write-S3Object -BucketName $BucketName -Key $s3Key -File $file.FullName
    }
    else {
        Write-Output "File $($file.Name) already exists in S3. Skipping upload."
    }
}

Write-Output "Upload process complete."
