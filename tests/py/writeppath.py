import sys
import os
mypath=sys.path[0]+'/../..'
sys.path.append(mypath)
#print(sys.path)
#print(os.getcwd())

fobj_out = open("ppath.py","w")
line="import sys\n"
fobj_out.write(line)
line="mypath='{}'\n".format(mypath)
fobj_out.write(line)
line="sys.path.append('{}')\n".format(mypath)
fobj_out.write(line)
fobj_out.close()
