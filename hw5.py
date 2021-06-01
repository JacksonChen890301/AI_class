import numpy as np

raw_image = np.array([[0,0,0,0,0,0,0,0,0,0],
                      [0,1,1,1,1,0,0,0,1,1],
                      [0,0,1,1,1,0,1,1,1,1],
                      [0,0,1,1,0,0,1,1,1,1],
                      [0,1,1,1,0,0,1,1,1,0],
                      [0,0,0,0,0,0,0,0,1,0],
                      [0,0,0,1,1,1,0,0,1,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,0,0,0,1,0,0,0]])
print(raw_image)

def segmentation(x, y, island):
    island[x, y] = 1
    raw_image[x, y] = 0
    for i in range(-1, 2, 2):
        try:
            if raw_image[x+i, y] == 1:
                segmentation(x+i, y, island)
        except IndexError:
            continue
    for i in range(-1, 2, 2):
        try:
            if raw_image[x, y+i] == 1:
                segmentation(x, y+i, island)
        except IndexError:
            continue
        



count = 0
for row in range(raw_image.shape[0]):
    for column in range(raw_image.shape[0]):
        if raw_image[row, column] == 1:
            island = np.zeros((10, 10))
            count+=1
            segmentation(row, column, island)
            print('island'+str(count)+':')
            print(island)
            print('size:', np.sum(island))





