f=open('/scratch/group/optmai/datasets/imagenet-a/README.txt',"r")
lines=f.readlines()
result=[]
for x in lines[12:-1]:
    result.append(x.split(' ')[1])
f.close()