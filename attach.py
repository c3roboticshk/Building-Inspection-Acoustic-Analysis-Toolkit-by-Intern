import cv2
import numpy as np
import os
import sys

# 定义基础路径和墙壁坐标字典
BASE_PATH = "..."
BG_PATH = '...'

#HKE parker cleaned coordinates

WALL_COORDINATES = {
    "south_wall01_s1": np.array([[22, 61], [31, 61], [31, 513], [22, 513]], dtype=np.float32),
    "south_wall01_s2": np.array([[32, 50], [63, 50], [63, 95], [31, 95]], dtype=np.float32),
    "south_wall01_s3": np.array([[32, 111], [63, 111], [63, 157], [32, 157]], dtype=np.float32),
"south_wall01_s4": np.array([[32, 173], [63, 173], [63, 218], [32, 218]], dtype=np.float32),
"south_wall01_s5": np.array([[32, 232], [63, 232], [63, 273], [32, 273]], dtype=np.float32),
"south_wall01_s6": np.array([[32, 289], [63, 289], [63, 330], [32, 330]], dtype=np.float32),
"south_wall01_s7": np.array([[32, 346], [63, 346], [63, 387], [32, 387]], dtype=np.float32),
"south_wall01_s8": np.array([[32, 403], [63, 403], [63, 447], [32, 447]], dtype=np.float32),
"south_wall01_s9": np.array([[32, 460], [63, 460], [63, 512], [32, 512]], dtype=np.float32),
    "south_wall02_s1":np.array([[64, 48], [173, 48], [173, 96], [64, 96]], dtype=np.float32),
    "south_wall02_s2": np.array([[179, 49], [270, 49], [270, 95], [179, 95]], dtype=np.float32),
    "south_wall03_s1": np.array([[66,110], [133, 110], [133, 156], [66, 156]], dtype=np.float32),
    "south_wall03_s2": np.array([[64, 158], [132, 158], [132, 216], [64, 216]], dtype=np.float32),
    "south_wall03_s3": np.array([[132, 108], [173, 108], [173, 216], [131, 216]], dtype=np.float32),
    "south_wall03_s4": np.array([[179, 108], [269, 108], [270, 215], [178, 215]], dtype=np.float32),
    "south_wall04_s1": np.array([[98, 245], [172, 244], [172, 274], [99, 274]], dtype=np.float32),
    "south_wall04_s2": np.array([[178, 243], [270, 242], [271, 271], [178, 272]], dtype=np.float32),
    "south_wall05_s1": np.array([[100, 289], [173, 289], [173, 332], [99, 332]], dtype=np.float32),
    "south_wall05_s2": np.array([[178, 292], [270, 292], [270, 331], [178, 332]], dtype=np.float32),
    "south_wall06_s1": np.array([[271, 47], [303, 48], [303, 332], [272, 332]], dtype=np.float32),
    "south_wall06_s2": np.array([[272, 332], [303, 332], [304, 507], [272, 507]], dtype=np.float32),
    "south_wall06_s3": np.array([[304, 47], [328, 47], [328, 89], [304, 89]], dtype=np.float32),
    "south_wall06_s4": np.array([[304, 101], [328, 101], [328, 153], [304, 153]], dtype=np.float32),
    "south_wall06_s5": np.array([[304, 167], [328, 167], [328, 207], [304, 207]], dtype=np.float32),
    "south_wall06_s6": np.array([[304, 219], [328, 219], [328, 269], [304, 269]], dtype=np.float32),
    "south_wall06_s7": np.array([[304, 282], [328, 282], [328, 332], [304, 332]], dtype=np.float32),
    "south_wall06_s8": np.array([[304, 342], [328, 342], [328, 386], [304, 386]], dtype=np.float32),
    "south_wall06_s9": np.array([[304, 397], [328, 397], [328, 446], [304, 446]], dtype=np.float32),
    "south_wall06_s10": np.array([[304, 459], [328, 459], [328, 506], [304, 506]], dtype=np.float32),
    "south_wall07_s1": np.array([[331, 48], [863, 44], [862, 88], [332, 89]], dtype=np.float32),
    "south_wall08_s1": np.array([[331, 109], [866, 101], [866, 211], [333, 216]], dtype=np.float32),
    "south_wall09_s1": np.array([[332, 231], [545, 231], [544, 272], [333, 276]], dtype=np.float32),
    "south_wall09_s2": np.array([[565, 231], [866, 229], [866, 268], [565, 273]], dtype=np.float32),
    "south_wall10_s1": np.array([[332, 292], [513, 291], [513, 331], [333, 333]], dtype=np.float32),
    "south_wall10_s2": np.array([[593, 291], [609, 291], [609, 328], [593, 328]], dtype=np.float32),
    "south_wall10_s3": np.array([[615, 290], [866, 289], [868, 326], [619, 328]], dtype=np.float32),
    "south_wall11_s1": np.array([[866, 44], [880, 44], [880, 328], [868, 327]], dtype=np.float32),
    "south_wall11_s2": np.array([[881, 44], [894, 44], [895, 270], [882, 272]], dtype=np.float32),
    "south_wall11_s3": np.array([[895, 44], [908, 44], [908, 327], [897, 328]], dtype=np.float32),
    "south_wall11_s4": np.array([[910, 44], [924, 44], [923, 320], [911, 320]], dtype=np.float32),
    "south_wall12_s1": np.array( [[926, 44], [1149, 39], [1149, 85], [930, 85]],dtype=np.float32),
    "south_wall13_s1": np.array([[932, 102], [1149, 103], [1150, 216], [933, 215]], dtype=np.float32),
    "south_wall14_s1": np.array([[931, 231], [1150, 231], [1149, 274], [933, 273]], dtype=np.float32),
    "south_wall15_s1": np.array([[932, 289], [1149, 288], [1149, 331], [933, 328]], dtype=np.float32),
    "south_wall16_s1": np.array([[1149, 14], [1162, 13], [1163, 286], [1147, 284]], dtype=np.float32),
    "south_wall16_s2": np.array([[1164, 13], [1187, 13], [1188, 85], [1165, 86]], dtype=np.float32),
    "south_wall16_s3": np.array( [[1164, 102], [1187, 101], [1187, 153], [1165, 154]],dtype=np.float32),
    "south_wall16_s4": np.array( [[1165, 171], [1188, 171], [1188, 214], [1165, 215]],dtype=np.float32),
    "south_wall16_s5": np.array([[1164, 227], [1187, 229], [1188, 270], [1166, 272]], dtype=np.float32),
    "south_wall16_s6": np.array([[1164, 288], [1188, 288], [1188, 331], [1166, 331]], dtype=np.float32),
    "south_wall16_s7": np.array([[1164, 348], [1185, 346], [1187, 389], [1162, 391]], dtype=np.float32),
    "south_wall16_s8": np.array([[1188, 14], [1196, 13], [1200, 406], [1188, 407]], dtype=np.float32),
    "south_wall17_s1": np.array( [[24, 9], [60, 9], [60, 49], [24, 49]],dtype=np.float32),
    "south_wall17_s2": np.array( [[67, 9], [80, 9], [80, 49], [67, 49]],dtype=np.float32),
    "south_wall17_s3": np.array( [[80, 9], [173, 9], [173, 49], [80, 49]],dtype=np.float32),
    "south_wall17_s4": np.array( [[182, 9], [319, 9], [319, 28], [182,28]],dtype=np.float32),
    "south_wall17_s5": np.array( [[258, 29], [288, 29], [288, 49], [258, 49]],dtype=np.float32),
    "south_wall17_s6": np.array( [[303, 29], [319, 29], [319, 49], [303, 49]],dtype=np.float32)
    # 添加更多墙面的坐标...
}
WALL_COORDINATES2={
    "east_s1": np.array( [[1019, 1206],[ 1515, 606],[ 1054, 4073],[93, 4059]],dtype=np.float32),
    "east_s2": np.array( [[1502, 737], [1922, 234], [1943, 303], [1495,820]],dtype=np.float32),
    "east_s3": np.array( [[1391, 1530], [2053, 896], [2080, 986], [1384, 1620]],dtype=np.float32),
    "east_s4": np.array( [(1371, 1792), (2115, 1116), (2797, 4073), (1088, 4066)],dtype=np.float32)
}
WALL_COORDINATES3={
    "west_s1(19)": np.array( [(438, 130), (559, 176), (578, 1592), (-113, 1595)],dtype=np.float32),
    "west_s2(20)": np.array( [(565, 181), (627, 208), (638, 314), (568, 284)],dtype=np.float32),
    "west_s3(win1)": np.array( [(565, 305), (641, 338), (649, 370), (570, 341)],dtype=np.float32),
    "west_s4(win2)": np.array( [(568, 373), (657, 408), (665, 451), (568, 419)],dtype=np.float32),
    "west_s5(win3)": np.array([(565, 454), (676, 489), (895, 1595), (524, 1597)], dtype=np.float32),
    "west_s6(21+22)": np.array([(632, 216), (724, 257), (1164, 1585), (892, 1597)], dtype=np.float32),
    "west_s7(23+24)": np.array([(741, 262), (884, 316), (1804, 1676), (1159, 1592)], dtype=np.float32)
}

# 图像缩放因子 (0-1之间，值越小图像越小)
SCALE_FACTOR = 1


def resize_image(img, scale_factor):
    """按比例缩放图像"""
    if scale_factor == 1.0:
        return img
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def project_images_to_background(wall_coords, base_path, scale_factor=1.0):
    # 加载背景图像
    bg_path = BG_PATH
    background = cv2.imread(bg_path)
    if background is None:
        raise FileNotFoundError(f"背景图像未找到: {bg_path}")

    print(f"原始背景图像尺寸: {background.shape[1]}x{background.shape[0]}")

    # 缩放背景图像
    if scale_factor != 1.0:
        print(f"缩放背景图像 (因子: {scale_factor})")
        background = resize_image(background, scale_factor)
        print(f"缩放后背景图像尺寸: {background.shape[1]}x{background.shape[0]}")

    # 创建背景副本用于绘制
    result = background.copy()

    for wall_id, coords in wall_coords.items():
        # 缩放坐标点
        scaled_coords = coords * scale_factor

        # 构建plot.png完整路径
        img_path = os.path.join(base_path, wall_id, 'plot.png')

        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: {img_path} 不存在，跳过")
            continue

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告: 无法读取 {img_path}，跳过")
            continue

        print(f"处理 {wall_id}: 图像尺寸 {img.shape[1]}x{img.shape[0]}")

        # 缩放输入图像
        if scale_factor != 1.0:
            img = resize_image(img, scale_factor)

        # 获取图像尺寸
        h, w = img.shape[:2]

        # 定义源点（原始图像的四个角点）
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        try:
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, scaled_coords)

            # 应用透视变换
            warped = cv2.warpPerspective(img, M, (background.shape[1], background.shape[0]))

            # 处理透明通道（如果存在）
            if warped.shape[2] == 4:  # 包含alpha通道
                # 分离颜色通道和alpha通道
                warped_rgb = warped[:, :, :3]
                alpha = warped[:, :, 3] / 255.0

                # 创建alpha通道的3D版本用于混合
                alpha_3d = alpha[:, :, np.newaxis]

                # 使用alpha混合
                result = (result * (1 - alpha_3d) + warped_rgb * alpha_3d).astype(np.uint8)
            else:
                # 创建掩码（非黑色区域）
                mask = np.any(warped > [0, 0, 0], axis=-1)
                result[mask] = warped[mask]

            print(f"成功投影 {wall_id}")

        except Exception as e:
            print(f"处理 {wall_id} 时出错: {str(e)}")
            # 打印坐标信息帮助调试
            print(f"源点: {src_points}")
            print(f"目标点: {scaled_coords}")
            print(f"背景尺寸: {background.shape[1]}x{background.shape[0]}")

    # 如果需要，缩放回原始尺寸
    if scale_factor != 1.0:
        print("缩放结果图像回原始尺寸...")
        result = resize_image(result, 1.0 / scale_factor)

    # 保存结果
    output_path = os.path.join(base_path, 'projected_building.jpg')
    cv2.imwrite(output_path, result)
    print(f"投影完成，结果已保存为 {output_path}")
    return result


# 执行投影
if __name__ == "__main__":
    try:
        # 确保基础路径存在
        if not os.path.exists(BASE_PATH):
            raise FileNotFoundError(f"基础路径不存在: {BASE_PATH}")

        print(f"开始处理路径: {BASE_PATH}")
        print(f"找到 {len(WALL_COORDINATES)} 个墙壁定义")

        # 尝试不同的缩放因子
        scale=1.0  # 从最小开始尝试
        result_image = project_images_to_background(WALL_COORDINATES, BASE_PATH, scale)

        # 显示结果（可选）
        cv2.imshow('Projected Building', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback

        traceback.print_exc()