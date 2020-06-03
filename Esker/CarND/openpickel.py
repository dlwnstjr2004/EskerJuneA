import pickle
import numpy as np
'''
list = []

with open ("classifier.p","rb") as f:
    while 1:
        try:
            temp_data = pickle.load(f)
        except EOFError:
            break
        list.append(temp_data)
'''
'''
with open('classifier.p', 'rb') as f:
    data = pickle.load(f)

scaler = data['scaler']
cls = data['classifier']
all_features = np.zeros((total_windows, self.num_features), dtype = np.float32)

fuck = scaler.transform(all_features)
print(fuck)
print(cls)
print(data)
'''

with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
perspective_data = pickle.load(f)

perspective_transform = perspective_data["perspective_transform"]
pixels_per_meter = perspective_data['pixels_per_meter']
orig_points = perspective_data["orig_points"]

print(str(perspective_transform))
print(str(pixels_per_meter))
print(str(orig_points))
