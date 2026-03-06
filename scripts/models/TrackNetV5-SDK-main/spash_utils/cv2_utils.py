import cv2


def get_opencv_version():
    version = cv2.__version__.split('.')
    major = int(version[0])
    minor = int(version[1])

    return major * 100 + minor  # return version like 410 for 4.10