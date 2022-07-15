import os
import time
import shutil

f = open('launch.sh', 'r')
words = f.readlines()[-1].split(' ')
f.close()
for word in words:
    if 'configs' in word:
        config = word.strip()
print('Load config from ', config)

f = open(config, 'r')
save_path = f.readlines()[-1].split(' ')[1].strip()
f.close()
print('Save code to ', save_path)

try:
    os.mkdir(save_path)
except Exception:
    pass
shutil.copytree(os.getcwd(), os.path.join(save_path, 'code'))
print('Start process')
os.chdir(os.path.join(save_path, 'code'))
os.system('sh launch.sh')
print('Done')