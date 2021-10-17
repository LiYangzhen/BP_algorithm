from test import display_test
from transform import convert_img_to_csv

IMG_DIR = r"C:\Users\LYZ\Documents\Code\BP_algorithm\test"  #在此处修改为测试图片所在的地址

convert_img_to_csv(IMG_DIR)  #将会在项目根目录生成test.csv文件

display_test()