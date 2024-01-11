# from matplotlib import pyplot as plt


# def MyAnimation(num_frame=200, root_path='imgs/', save_path='cosxt.gif'):
#     # https://zhuanlan.zhihu.com/p/106283237
#     import matplotlib.animation as animation
#     import cv2

#     fig = plt.figure()
#     ims = []
#     for i in range(num_frame):
#         # 用opencv读取图片
#         img = cv2.imread(root_path+str(i+1)+'.png')
#         (r, g, b) = cv2.split(img)
#         img = cv2.merge([b, g, r])
#         im = plt.imshow(img, animated=True)
#         plt.axis('off')
#         # plt.show()
#         ims.append([im])
#     # 用animation中的ArtistAnimation实现动画. 每帧之间间隔500毫秒, 每隔1000毫秒重复一次,循环播放动画.
#     ani = animation.ArtistAnimation(
#         fig, ims, interval=1000, blit=True, repeat_delay=2000)

#     # 保存动态图片
#     ani.save(save_path, fps=10)


# if __name__ == '__main__':
#     MyAnimation(num_frame=200, root_path='imgs/', save_path='path.gif')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2


def MyAnimation(num_frame=200, root_path='imgs/', save_path='cosxt.gif', dpi=100):
    # 读取第一张图片来获取图像尺寸
    first_image = cv2.imread(root_path + '1.png')
    height, width, _ = first_image.shape
    # 计算fig的尺寸（英寸）
    fig_width = width / dpi
    fig_height = height / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ims = []
    for i in range(num_frame):
        img = cv2.imread(root_path + str(i+1) + '.png')
        (r, g, b) = cv2.split(img)
        img = cv2.merge([b, g, r])
        im = plt.imshow(img, animated=True)
        plt.axis('off')
        ims.append([im])

    plt.tight_layout()  # 确保布局紧凑

    ani = animation.ArtistAnimation(
        fig, ims, interval=200, blit=True, repeat_delay=2000)

    ani.save(save_path, writer='imagemagick', dpi=dpi)  # 保存GIF


if __name__ == '__main__':
    MyAnimation(num_frame=200, root_path='imgs/',
                save_path='gifs/8.gif', dpi=100)
