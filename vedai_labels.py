import os

def yolo_annot(root, dest, w_h_ratio):
    for r,d,c in os.walk(root):
        values = c

    i =0 
    for f in values:
        result = os.path.join(root,f)
        print(result)
        with open(result, 'r') as temp:
            content = temp.readlines()
            dest_pos = os.path.join(dest,f[:-4] + "_co.txt")
            with open(dest_pos, 'w') as d:
                for v in content:
                    line = v.split(' ')
                    if(int(line[3]) == 1):
                        class_id = 0
                    elif(int(line[3]) == 2):
                        class_id = 1
                    elif(int(line[3]) == 4):
                        class_id = 2
                    elif(int(line[3]) == 5):
                        class_id = 3
                    elif(int(line[3]) == 9):
                        class_id = 4
                    elif(int(line[3]) == 11):
                        class_id = 5
                    elif(int(line[3]) == 23):
                        class_id = 6
                    elif(int(line[3]) == 31):
                        class_id = 7
                    elif(int(line[3]) == 10):
                        class_id = 8
                    else:
                        print(line[3])
                        print(f)
                        print(line)
                        class_id = 8
                    xmin = min(float(line[6]), float(line[9]), float(line[7]), float(line[8])) * w_h_ratio
                    xmax = max(float(line[6]), float(line[9]), float(line[7]), float(line[8])) * w_h_ratio
                    ymin = min(float(line[10]), float(line[11]), float(line[12]), float(line[13])) * w_h_ratio
                    ymax = max(float(line[10]), float(line[11]), float(line[12]), float(line[13])) * w_h_ratio
                    box = (float(line[0]) * w_h_ratio, float(line[1]) * w_h_ratio, float(xmax - xmin), float(ymax - ymin))
                    if(box[2] < 0 or box[3] < 0):
                        print(f)
                        print(box)
                    d.write(str(class_id) + " " + " ".join([str(a) for a in box]) + "\n")
                    i = i+1
    return(-1)
                        

def main():
    root = "/scratch/ramrao/vehicles/Annotations512/"
    dest = "/scratch/ramrao/vehicles/labels/"
    resized = 448
    image_size = 512
    w_h_ratio = resized / image_size
    val = yolo_annot(root, dest, w_h_ratio)
    

if(__name__ == "__main__"):
    main()
