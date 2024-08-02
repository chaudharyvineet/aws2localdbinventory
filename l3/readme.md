1. upload file to new bucket
     -- lambda_function.zip
     -- layer.zip

2. create function use lambda_function.zip's s3 uri { use vpc }
4. create layer -- use layer.zip
5. create efs then a access point in the same region and vpc as lambda -- with all values as 1000, and perm 777 -- look at annx1 below
6. add file system to lambda and run

7. policy for efs, choose aws role as principal

--- annx1 --- 
POSIX user
User ID
1000
Group ID
1000
Secondary group IDs
-
Root directory creation permissions
Owner user ID
1000
Owner group ID
1000
Permissions
777
Root directory path
/efs
--

   \test
