import pandas as pd
import os
from PIL import Image

def generate_csv():
    ROOTS = '../Dataset/'
    PHASE = ['train', 'val']
    CLASSES = ['Mammals', 'Birds']  # [0,1]
    SPECIES = ['rabbits', 'rats', 'chickens']  # [0,1,2]

    # SPECIES = ['rabbits', 'rats', 'chickens','Mammals', 'Birds']  # [0,1,2,3,4]

    DATA_info = {'train': {'path': [], 'species': [], 'classes': []},
                 'val': {'path': [], 'species': [], 'classes': []}
                 }
    for p in PHASE:
        for s in SPECIES:
            DATA_DIR = ROOTS + p + '/' + s
            DATA_NAME = os.listdir(DATA_DIR)

            count = 0
            for item in DATA_NAME:
                count += 1
                #debug 调试阶段，样本减少15倍
    #             if count%15 == 1:
                try:
                    img = Image.open(os.path.join(DATA_DIR, item))
                except OSError:
                    pass
                else:
                    DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                    if s == 'rabbits':
                        DATA_info[p]['species'].append(0)
                        DATA_info[p]['classes'].append(0)
                    elif s == 'rats':
                        DATA_info[p]['species'].append(1)
                        DATA_info[p]['classes'].append(0)
                    else:
                        DATA_info[p]['species'].append(2)
                        DATA_info[p]['classes'].append(1)

        ANNOTATION = pd.DataFrame(DATA_info[p])
        ANNOTATION.to_csv('Multi_%s_annotation.csv' % p)
        print('Multi_%s_annotation file is saved.' % p)


class ImageRename():
    def __init__(self, r, s):
        self.path = r + '/' + s
        self.s = s

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        n = 0
        for item in filelist:
            if item.endswith('.jpg'):
                oldname = os.path.join(os.path.abspath(self.path), item)
                newname = os.path.join(os.path.abspath(self.path), self.s + format(str(n), '0>3s') + '.jpg')

                os.rename(oldname, newname)
                n += 1
        print('total %d to rename & converted %d jpgs' % (total_num, n))


def image_rename():
    PHASE = ['../Dataset/train', '../Dataset/val']
    SPECIES = ['rabbits', 'rats', 'chickens']
    for p in PHASE:
        for s in SPECIES:
            newname = ImageRename(p, s)
            newname.rename()