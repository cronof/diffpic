import sys # to access the system
import cv2
from PIL import Image

# img = Image.open('./myimage.png')
# img.show()




def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def DoG(img1, img2, ksize, sigma1, sigma2):
    # 標準偏差が異なる2つのガウシアン画像を算出
    g1 = cv2.GaussianBlur(img1, ksize, sigma1)
    g2 = cv2.GaussianBlur(img2, ksize, sigma2)
    # 2つのガウシアン画像の差分を出力
    return g1 - g2
 

# def main():
#     # 入力画像を読み込み
#     img = cv2.imread("input.jpg")

#     # グレースケール変換
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)





def main():
    img = cv2.imread("image01.png", cv2.IMREAD_ANYCOLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    
    h, w = img.shape[:2]
    img1 = img[0:int(h), 0:int(w/2)]
    img2 = img[0:int(h), int(w/2):int(w)]

    # DoGフィルタ処理
    # dst = DoG(img1, img2, (3,3), 1.3, 2.6)

    contours1, hierarchy1 = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours2, hierarchy2 = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    result = cv2.drawContours(gray, contours1, -1, (0,255,0), 3)
    # dst = img1 - img2

    # 結果を出力
    # cv2.imwrite("img1.png", img1 )
    # cv2.imwrite("img2.png", img2 )
    cv2.imwrite("result.png", result )
    


    # while True:
    #     cv2.imshow("Sheep", cropped_image)
    #     cv2.waitKey(3000)
    #     sys.exit() # to exit from all the processes
    
    # cv2.destroyAllWindows() # destroy all windows


if __name__ == "__main__":
    main()
        
