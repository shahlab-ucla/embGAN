from stardist.models import StarDist2D
from PIL import Image
import numpy as np
import os

model = StarDist2D.from_pretrained('2D_versatile_fluo')

dic_dir = './DIC'
fluo_dir = './fluo'
out_dir = './labels'

for f in sorted(os.listdir(fluo_dir)):
    fluo = np.array(Image.open(fluo_dir + f))
    labels, rtn_dict = model.predict_instances(normalize(fluo))
    labels[labels>=1] = 255
    labels = labels.astype('uint8')
    labels = Image.fromarray(labels)
    labels.save(os.path.join(out_dir,f))    