# Copyright 2019 ModelArts Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from PIL import Image
from model_service.tfserving_model_service import TfServingBaseService
IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
MODEL_OUTPUT_KEY = 'logits'
LABELS_FILE_NAME = 'small_labels_25c.txt'
def decode_image(file_content):
"""
Decode bytes to a single image
:param file_content: bytes
:return: ndarray with rank=3
"""
image = Image.open(file_content)
image = image.convert('RGB')
image = np.asarray(image, dtype=np.float32)
return image
def read_label_list(path):
"""
read label list from path
:param path: a path
:return: a list of label names like: ['label_a', 'label_b', ...]
"""
with open(path, 'r') as f:
label_list = f.read().split(os.linesep)
label_list = [x.strip() for x in label_list if x.strip()]
return label_list
class FoodPredictService(TfServingBaseService):

def _preprocess(self, data):
"""
`data` is provided by Upredict service according to the input data. Which is
like:
{'images': {'image_a.jpg': b'xxx'}}
For now, predict a single image at a time.
"""
images = []
for file_name, file_content in data[IMAGES_KEY].items():
print('\tAppending image: %s' % file_name)
images.append(decode_image(file_content))
preprocessed_data = {MODEL_INPUT_KEY: np.asarray(images)}
return preprocessed_data
def _postprocess(self, data):
"""
`data` is the result of your model. Which is like:
{'logits': [[0.1, -0.12, 0.72, ...]]}
value of logits is a single list of list because one image is predicted at a
time for now.
"""
# label_list = ['label_a', 'label_b', 'label_c', ...]
label_list = read_label_list(os.path.join(self.model_path, LABELS_FILE_NAME))
# logits_list = [0.1, -0.12, 0.72, ...]
logits_list = data[MODEL_OUTPUT_KEY][0]
# labels_to_logits = {'label_a': 0.1, 'label_b': -0.12, 'label_c': 0.72, ...}
labels_to_logits = {label_list[i]: s for i, s in enumerate(logits_list)}
predict_result = {MODEL_OUTPUT_KEY: labels_to_logits}
return predict_result
