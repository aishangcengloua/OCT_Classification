import cv2
import numpy as np
import math
from pathlib import Path

def dataset_size(path) :
    parents_dirs = [name for name in Path(path).iterdir() if name.is_dir()]
    for dir in parents_dirs:
        total_numbers = 0
        for classes in dir.iterdir():
            total_numbers += 1
        print(total_numbers)

class ImagePreprocess :
    def __init__(self, data_path, new_data_path):
        self.data_path = data_path
        self.new_data_path = Path(new_data_path)
        self.get_items()

    def get_items(self):
        classes_paths = [name.__str__() for name in Path(self.data_path).iterdir() if name.is_dir()]
        classes_paths.sort()
        classes_to_idxs = {classes_paths[i].split('\\')[-1]: i for i in range(len(classes_paths))}
        self.items = []
        for idx, target in enumerate(classes_paths):
            if not Path(target).is_dir():
                continue
            for image_path in Path(target).iterdir():
                fname = image_path.__str__().split('.')[-1]
                if fname == 'tif' or fname == 'jpeg' or fname == 'png' or fname == 'jpg':
                    self.items.append(image_path.__str__())
        self.items.sort()
        self.num_items = len(self.items)

    def preprocess(self):
        for img_path in self.items :
            path_split = img_path.split('\\')
            img_name = path_split[-1]
            img_parents_path = "/".join(path_split[-3: -1])
            new_img_path = img_parents_path + '/'
            save_path = self.new_data_path.joinpath(new_img_path)
            if not save_path.exists():
                save_path.mkdir(parents = True, exist_ok = False)

            if save_path.joinpath(img_name).exists() :
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_brg = cv2.imread(img_path, cv2.IMREAD_COLOR)
            original = self.gauss_filter(img_brg)

            img = self.gauss_filter(img)
            img = self.binaryzation(img)
            img = self.media_fliter(img)
            img = self.morph(img).astype(np.uint8)
            parameter = self.fitting(img)
            img_process = self.normalization(img, original, parameter)

            cv2.imwrite(save_path.joinpath(img_name).__str__(), img_process)

    def gauss_filter(self, img):
        gauss = cv2.GaussianBlur(img, (5, 5), 0)
        return gauss

    def binaryzation(self, img):
        mean = np.mean(img)
        ret, binary = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)
        return binary

    def media_fliter(self, img):
        media = cv2.medianBlur(img, 7)
        contours, hierarchy = cv2.findContours(media, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(img.shape)
        area = []
        for idx in range(len(contours)) :
            area.append(cv2.contourArea(contours[idx]))

        # max_area = cv2.contourArea(contours[np.argmax(area)])
        area = np.array(area)
        result = cv2.fillPoly(mask, [contours[np.argmax(area)]], 255)
        return result

    def morph(self, img):
        kernel = np.ones((5, 5), np.uint8)
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((25, 25), np.uint8)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
        return close

    def linear_fitting(self, x, y):
        parameter = np.polyfit(x, y, 1)
        function = np.poly1d(parameter)
        return parameter, function

    def polynomial_fitting(self, x, y):
        parameter = np.polyfit(x, y, 2)
        function = np.poly1d(parameter)
        return parameter, function

    def image_parsing(self, img):
        img = np.array(img)
        height, width = img.shape

        x = np.zeros((1, width))
        ymid = np.zeros((1, width))
        ymin = np.zeros((1, width))
        for i in range(width):
            x[0][i] = i
            count = 0
            sum = 0
            for j in range(height):
                if img[j][i] == 255:
                    count += 1
                    sum += j
                elif img[j][i] == 0 and j > 0 and img[j - 1][i] == 255:
                    ymin[0][i] = j + 1
                    break
            if count == 0 and i != 0:
                ymid[0][i] = ymid[0][i - 1]
            elif count == 0:
                ymid[0][i] = height / 2
            else:
                ymid[0][i] = sum / count

        return x, ymid, ymin

    def coefficient(self, X, Y, function):
        size = X.size
        residual = 0
        total = 0
        for i in range(0, size - 1):
            y = function(X[i])
            residual += math.pow((y - Y[i]), 2)
            total += math.pow(y, 2)

        return (total - residual) / total

    def fitting(self, img):
        X, Ymid, Ymin = self.image_parsing(img)
        parameter, function = self.polynomial_fitting(X[0], Ymid[0])
        if parameter[0] > 0:
            return parameter
        else:
            parameter_polynomial, function_polynomial = self.polynomial_fitting(X[0], Ymin[0])
            parameter_linear, function_linear = self.linear_fitting(X[0], Ymin[0])
            if parameter_polynomial[0] > 0:
                c_polynomial = self.coefficient(X[0], Ymin[0], function_polynomial)
                c_linear = self.coefficient(X[0], Ymin[0], function_linear)
                if c_polynomial > c_linear:
                    return parameter_polynomial
                else:
                    return parameter_linear
            else:
                return parameter_linear

    def adjust(self, img, width, height, function, minvalue):
        for i in range(width):
            move = int(function(i) - minvalue)
            for j in range(height - 1, move, -1):
                img[j][i] = img[j - move][i]

            move = move if move < height else height - 1
            for j in range(move, -1, -1):
                img[j][i] = 0
        return img

    def tailoring(self, img, width, height):
        minheight = 0
        maxheight = height - 1
        i = 0
        j = 0
        for j in range(height):
            for i in range(width):
                if img[j][i] == 255:
                    minheight = j
                    break
            if i < width - 1:
                break

        for j in range(height - 1, -1, -1):
            for i in range(width):
                if img[j][i] == 255:
                    maxheight = j
                    break
            if i < width - 1:
                break
        return minheight, maxheight

    def normalization(self, img, img2, parameter):
        height, width = img.shape
        height2, width2, _ = img2.shape
        function = np.poly1d(parameter)
        if len(parameter) == 2:
            minvalue = min(function(0), function(width - 1))
        else:
            if parameter[0] > 0:
                if 0 < -parameter[1] / (2 * parameter[0]) < width:
                    minvalue = min(function(0), function(width - 1),
                                   (4 * parameter[0] * parameter[2] - parameter[1] * parameter[1]) / (4 * parameter[0]))
                else:
                    minvalue = min(function(0), function(width - 1))
            else :
                minvalue = min(function(0), function(width - 1))

        img = self.adjust(img, width, height, function, minvalue)
        minheight, maxheight = self.tailoring(img, width, height)
        img = np.delete(img, np.arange(maxheight, height - 1, 1), axis = 0)
        img = np.delete(img, np.arange(0, minheight - 1, 1), axis = 0)

        b, g, r = cv2.split(img2)
        b = self.adjust(b, width2, height2, function, minvalue)
        g = self.adjust(g, width2, height2, function, minvalue)
        r = self.adjust(r, width2, height2, function, minvalue)
        img2 = cv2.merge([b, g, r])
        img2 = np.delete(img2, np.arange(maxheight, height2 - 1, 1), axis = 0)
        img2 = np.delete(img2, np.arange(0, minheight - 1, 1), axis = 0)

        return img2

if __name__ == '__main__':
    processer = ImagePreprocess('../OCT_Dataset/OCT2017/train/', '../OCT_Dataset/Processed_OCT2017')
    processer.preprocess()

    processer = ImagePreprocess('../OCT_Dataset/OCT2017/test/', '../OCT_Dataset/Processed_OCT2017')
    processer.preprocess()