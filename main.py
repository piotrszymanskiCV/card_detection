
import imutils
import cv2

import numpy as np
from skimage.measure import compare_ssim


def znajdz_karte(imgoutput):
    resized, normal = figura(imgoutput)
    cropped2 = figura_prostakat(resized, normal)

    FINAL = get_comp(cropped2)
    cv2.waitKey(0)
    max = 0.0
    for test in testery:
        image12 = cv2.imread(test)
        a = cv2.resize(image12, (300, 200))
        b = cv2.resize(FINAL, (300, 200))
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, b, full=True)
        diff = (diff * 255).astype(int)
        if (score > max):
            max = score
            wynik = test

        print("Structural Similarity Index: {}".format(score))
    print(max, wynik)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imgoutput, 'OpenCV', (10, 500), font, 4, (255, 0, 255), 2, cv2.LINE_AA)
    image = cv2.resize(imgoutput, (600, 900))


def check_card(image1):
    resized, normal = figura(image1)
    cropped2 = figura_prostakat(resized, normal)

    FINAL = get_comp(cropped2)
    cv2.waitKey(0)
    max = 0.0
    for test in testery.keys():
        image12 = cv2.imread(test)
        a = cv2.resize(image12, (300, 200))
        b = cv2.resize(FINAL, (300, 200))
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, b, full=True)
        diff = (diff * 255).astype(int)
        if (score > max):
            max = score
            wynik = testery[test]

        print("Structural Similarity Index: {}".format(score))

    font = cv2.FONT_HERSHEY_SIMPLEX
    return max, wynik


def perspektywa(image, c):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    cv2.drawContours(image, [c], -1, (0, 255, 255), 5)

    cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    cv2.circle(image, extBot, 8, (255, 255, 0), -1)
    dist = (extBot[0] - extLeft[0]) ** 2 + (extBot[0] - extLeft[0]) ** 2
    dist2 = (extTop[0] - extBot[0]) ** 2 + (extTop[0] - extBot[0]) ** 2


    pts1 = np.float32([list(extLeft), list(extTop), list(extBot), list(extRight)])
    pts2 = np.float32([[250, 400], [0, 400], [250, 0], [0, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    print("1")

    imgoutput = cv2.warpPerspective(image, matrix, (250, 400))
    return imgoutput


def karta(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    good = []
    for tt in cnts:
        area = cv2.contourArea(tt)
        if area > 10000:
            good.append(tt)
    ww = []
    ss = []
    for c in good:

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        ss.append((cX, cY))
        ww.append(perspektywa(image, c))
    return (ww, ss)


def figura(imgoutput):
    pp = imgoutput.copy()
    gray1 = cv2.cvtColor(imgoutput, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray1 = cv2.Canny(gray1, 25, 200)

    thresh1 = cv2.threshold(gray1, 180, 255, cv2.THRESH_BINARY)[1]

    cv2.waitKey(0)

    pp = pp[5:70, 0:45]
    cropped = thresh1[5:70, 0:45]
    cropped = cv2.resize(cropped, (200, 240), interpolation=cv2.INTER_AREA)
    pp = cv2.resize(pp, (200, 240), interpolation=cv2.INTER_AREA)

    cv2.imshow("hhhh", pp)
    cv2.waitKey(0)
    ww, kk = cv2.findContours(cropped.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    good = []
    for tt in ww:
        area = cv2.contourArea(tt)
        if area > 100:
            good.append(tt)

    c1 = max(ww, key=cv2.contourArea)
    # cv2.drawContours(cropped, [c1], -1, (250, 0, 250), 3)

    # resized2 = cv2.resize(cropped, (360, 480), interpolation=cv2.INTER_AREA)

    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # cv2.drawContours(resized, cnts1, -1, (0, 255, 0), 2)

    return cropped, pp


def figura_prostakat(resized, normal):
    cnts1, ww = cv2.findContours(resized.copy(), cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_SIMPLE)

    ################33
    good = []
    for tt in cnts1:
        area = cv2.contourArea(tt)
        if area > 0:
            good.append(tt)
    c = max(good, key=cv2.contourArea)

    cv2.drawContours(resized, [c], 0, (0, 255, 0), 3)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    # cv2.rectangle(resized, (x,y), (x+w , y+h),(0,255,0),3)
    print(x, y, w, h)
    cropped2 = normal[y:(y + h), x:(w + x)]

    return cropped2


def test(image):
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)


    thresh1 = cv2.threshold(gray1, 165, 255, cv2.THRESH_BINARY)[1]


def get_comp(imgare):
    gray1 = cv2.cvtColor(imgare, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)


    thresh1 = cv2.threshold(gray1, 165, 255, cv2.THRESH_BINARY)[1]

    return thresh1


def check_znak(image):
    pp = image.copy()
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray1 = cv2.Canny(gray1, 25, 200)

    thresh1 = cv2.threshold(gray1, 180, 255, cv2.THRESH_BINARY)[1]

    cv2.waitKey(0)

    pp = pp[50:110, 0:40]
    cropped = thresh1[50:110, 0:40]
    cropped = cv2.resize(cropped, (200, 240), interpolation=cv2.INTER_AREA)
    pp = cv2.resize(pp, (200, 240), interpolation=cv2.INTER_AREA)

    cv2.waitKey(0)
    ww, kk = cv2.findContours(cropped.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    good = []
    for tt in ww:
        area = cv2.contourArea(tt)
        if area > 100:
            good.append(tt)

    c1 = max(ww, key=cv2.contourArea)
    # cv2.drawContours(cropped, [c1], -1, (250, 0, 250), 3)

    # resized2 = cv2.resize(cropped, (360, 480), interpolation=cv2.INTER_AREA)

    zw = figura_prostakat(cropped, pp)

    zw = get_comp(zw)

    # cv2.imwrite("C:/Users/48781/Desktop/znaki/dzwon.jpg",zw)

    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # cv2.drawContours(resized, cnts1, -1, (0, 255, 0), 2)

    return zw


def wynik_znak(image1):
    FINAL = image1
    cv2.waitKey(0)
    max = 0.0
    for test in znaki.keys():
        image12 = cv2.imread(test)
        a = cv2.resize(image12, (300, 200))
        b = cv2.resize(FINAL, (300, 200))
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, b, full=True)
        diff = (diff * 255).astype(int)
        if (score > max):
            max = score
            wynik = znaki[test]

    font = cv2.FONT_HERSHEY_SIMPLEX
    return max, wynik


easy = ["C:/Users/48781/Desktop/medium/12.jpg",
        "C:/Users/48781/Desktop/easy/1.jpg",
        "C:/Users/48781/Desktop/easy/2.jpg",
        "C:/Users/48781/Desktop/easy/3.jpg",
        "C:/Users/48781/Desktop/easy/4.jpg",
        "C:/Users/48781/Desktop/easy/5.jpg",
        "C:/Users/48781/Desktop/easy/6.jpg",
        "C:/Users/48781/Desktop/easy/7.jpg",
        "C:/Users/48781/Desktop/easy/8.jpg",
        "C:/Users/48781/Desktop/easy/9.jpg",
        "C:/Users/48781/Desktop/medium/10.jpg",
        "C:/Users/48781/Desktop/medium/11.jpg",
        "C:/Users/48781/Desktop/medium/13.jpg",
        "C:/Users/48781/Desktop/medium/14.jpg",
        ]

image1 = cv2.imread("C:/Users/48781/Desktop/oseim_test2.jpg")
testery = {
    "C:/Users/48781/Desktop/comparer/2.jpg": "DWOJKA",
           "C:/Users/48781/Desktop/comparer/3.jpg": "TROJKA",
           "C:/Users/48781/Desktop/comparer/4.jpg": "CZWORKA",
           "C:/Users/48781/Desktop/comparer/5.jpg": "PIATKA",
           "C:/Users/48781/Desktop/comparer/6.jpg": "SZOSTKA",
           "C:/Users/48781/Desktop/comparer/7.jpg": "SIODEMKA",
           "C:/Users/48781/Desktop/comparer/8.jpg": "OSEMKA",
           "C:/Users/48781/Desktop/comparer/9.jpg": "DZIEWIATKA",
           "C:/Users/48781/Desktop/comparer/10.jpg": "DZIESIATKA",
           "C:/Users/48781/Desktop/comparer/J.jpg": "JOPEK",
           "C:/Users/48781/Desktop/comparer/K.jpg": "KROL",
           "C:/Users/48781/Desktop/comparer/Q.jpg": "DAMA",
           "C:/Users/48781/Desktop/comparer/as.jpg": "AS"}

znaki = {"C:/Users/48781/Desktop/znaki/zoladz.jpg": "ZOLADZ",
         "C:/Users/48781/Desktop/znaki/wino.jpg": "WINO",
         "C:/Users/48781/Desktop/znaki/dzwon.jpg": "DZWON",
         "C:/Users/48781/Desktop/znaki/serce.jpg": "SERCE", }
k=0;
for img in easy:
    image = cv2.imread(img)
    karty, srodki = karta(image)
    oo = "r"
    i = 0
    for k in karty:
        cv2.imshow(oo, k)
        cv2.waitKey(0)
        oo = oo + "r"
        prob, card = check_card(k)

        tt, znakk = wynik_znak(check_znak(k))

        cv2.putText(image, card, (srodki[i][0] - 300, srodki[i][1] - 300),
                    4, 3, (255, 0, 255), 15)
        cv2.putText(image, znakk, (srodki[i][0] - 300, srodki[i][1]),
                    4, 3, (255, 0, 255), 15)
        i = i + 1

    image = cv2.resize(image, (1000, 1200))

    cv2.imshow("final", image)
    cv2.imwrite("C:/Users/48781/Desktop/wyniki/kekw.jpg",image)
    k=k+1;
    cv2.waitKey(0)
