import cv2
import numpy as np
import csv



def line_intersection(a, b, c, d):

    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    # 计算方向向量
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # 计算分母
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:  # 平行或重合
        return None  # 假设总有交点，但返回None时需处理

    # 计算参数t和s
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    # 计算交点
    x = x1 + t * dx1
    y = y1 + t * dy1
    return (x, y)


def process_coordinates(points):
    points=invert_y_coordinates(points)

    n = len(points)
    if n < 4:
        raise ValueError("坐标数量必须至少为4")

    while n > 4:
        changed = False  # 标记本轮是否修改了列表
        i = 0
        while i < n and not changed:
            # 当前向量：从 points[i] 到 points[(i+1) % n]
            p_i = points[i]
            p_i1 = points[(i + 1) % n]
            p_i2 = points[(i + 2) % n]
            dx_i = p_i1[0] - p_i[0]
            dy_i = p_i1[1] - p_i[1]
            dx_next = p_i2[0] - p_i1[0]
            dy_next = p_i2[1] - p_i1[1]

            if i % 4 == 0:
                if dx_i < 0:
                    if dy_next<0:
                        points.pop((i + 2) % n)
                        points.pop((i + 1) % n)
                        points.pop(i)
                        points.pop((i - 1) % n)
                        n = len(points)
                    else:
                        # 条件2为true：进行交点操作
                        # 先存储相关点（基于原始索引）
                        p_i_minus2 = points[(i - 2) % n]  # V_{i-2} 起始点
                        p_i_minus1 = points[(i - 1) % n]  # V_{i-2} 终点，也是要删除的点
                        p_i_plus1 = points[(i + 1) % n]  # V_{i+1} 起始点，也是要删除的点
                        p_i_plus2 = points[(i + 2) % n]  # V_{i+1} 终点

                        # 删除第i个点（第i个向量的起始点）
                        points.pop(i)

                        line1_p1 = p_i_minus2
                        line1_p2 = p_i_minus1
                        # 对于反向延长，使用点 p_i_plus1 和 p_i_plus1 - (p_i_plus2 - p_i_plus1) = 2*p_i_plus1 - p_i_plus2
                        rev_point = (2 * p_i_plus1[0] - p_i_plus2[0], 2 * p_i_plus1[1] - p_i_plus2[1])
                        line2_p1 = p_i_plus1
                        line2_p2 = rev_point

                        intersection = line_intersection(line1_p1, line1_p2, line2_p1, line2_p2)
                        if intersection is None:
                            # 如果平行，添加中点作为fallback（根据问题假设应有交点，但安全处理）
                            intersection = ((line1_p1[0] + line2_p1[0]) / 2, (line1_p1[1] + line2_p1[1]) / 2)


                        points.insert(i,intersection)
                        points.pop((i + 1) % n)
                        points.pop(i - 1)
                        n=len(points)

                else:
                    # 条件1为true，检查下一个i
                    i += 1

            elif i % 4 == 1:
                if dy_i > 0:
                    if dx_next < 0:
                        points.pop((i + 2) % n)
                        points.pop((i + 1) % n)
                        points.pop(i)
                        points.pop((i - 1) % n)
                        n = len(points)
                    else:
                        # 条件2为true：进行交点操作
                        # 先存储相关点（基于原始索引）
                        p_i_minus2 = points[(i - 2) % n]  # V_{i-2} 起始点
                        p_i_minus1 = points[(i - 1) % n]  # V_{i-2} 终点，也是要删除的点
                        p_i_plus1 = points[(i + 1) % n]  # V_{i+1} 起始点，也是要删除的点
                        p_i_plus2 = points[(i + 2) % n]  # V_{i+1} 终点

                        # 删除第i个点（第i个向量的起始点）
                        points.pop(i)

                        line1_p1 = p_i_minus2
                        line1_p2 = p_i_minus1
                        # 对于反向延长，使用点 p_i_plus1 和 p_i_plus1 - (p_i_plus2 - p_i_plus1) = 2*p_i_plus1 - p_i_plus2
                        rev_point = (2 * p_i_plus1[0] - p_i_plus2[0], 2 * p_i_plus1[1] - p_i_plus2[1])
                        line2_p1 = p_i_plus1
                        line2_p2 = rev_point

                        intersection = line_intersection(line1_p1, line1_p2, line2_p1, line2_p2)
                        if intersection is None:
                            # 如果平行，添加中点作为fallback（根据问题假设应有交点，但安全处理）
                            intersection = (int((line1_p1[0] + line2_p1[0]) / 2), int((line1_p1[1] + line2_p1[1]) / 2))

                        points.insert(i, intersection)
                        points.pop((i + 1) % n)
                        points.pop(i - 1)
                        n = len(points)

                else:
                    # 条件1为true，检查下一个i
                    i += 1
            elif i % 4 == 2:
                if dx_i > 0:
                    if dy_next > 0:
                        points.pop((i + 2) % n)
                        points.pop((i + 1) % n)
                        points.pop(i)
                        points.pop((i - 1) % n)
                        n = len(points)
                    else:
                        # 条件2为true：进行交点操作
                        # 先存储相关点（基于原始索引）
                        p_i_minus2 = points[(i - 2) % n]  # V_{i-2} 起始点
                        p_i_minus1 = points[(i - 1) % n]  # V_{i-2} 终点，也是要删除的点
                        p_i_plus1 = points[(i + 1) % n]  # V_{i+1} 起始点，也是要删除的点
                        p_i_plus2 = points[(i + 2) % n]  # V_{i+1} 终点

                        # 删除第i个点（第i个向量的起始点）
                        points.pop(i)

                        line1_p1 = p_i_minus2
                        line1_p2 = p_i_minus1
                        # 对于反向延长，使用点 p_i_plus1 和 p_i_plus1 - (p_i_plus2 - p_i_plus1) = 2*p_i_plus1 - p_i_plus2
                        rev_point = (2 * p_i_plus1[0] - p_i_plus2[0], 2 * p_i_plus1[1] - p_i_plus2[1])
                        line2_p1 = p_i_plus1
                        line2_p2 = rev_point

                        intersection = line_intersection(line1_p1, line1_p2, line2_p1, line2_p2)
                        if intersection is None:
                            # 如果平行，添加中点作为fallback（根据问题假设应有交点，但安全处理）
                            intersection = ((line1_p1[0] + line2_p1[0]) / 2, (line1_p1[1] + line2_p1[1]) / 2)

                        points.insert(i, intersection)
                        points.pop((i + 1) % n)
                        points.pop(i - 1)
                        n = len(points)

                else:
                    # 条件1为true，检查下一个i
                    i += 1
            elif i % 4 == 3:
                if dy_i < 0:
                    if dx_next > 0:
                        points.pop((i + 2) % n)
                        points.pop((i + 1) % n)
                        points.pop(i)
                        points.pop((i - 1) % n)
                        n = len(points)
                    else:
                        # 条件2为true：进行交点操作
                        # 先存储相关点（基于原始索引）
                        p_i_minus2 = points[(i - 2) % n]  # V_{i-2} 起始点
                        p_i_minus1 = points[(i - 1) % n]  # V_{i-2} 终点，也是要删除的点
                        p_i_plus1 = points[(i + 1) % n]  # V_{i+1} 起始点，也是要删除的点
                        p_i_plus2 = points[(i + 2) % n]  # V_{i+1} 终点

                        # 删除第i个点（第i个向量的起始点）
                        points.pop(i)

                        line1_p1 = p_i_minus2
                        line1_p2 = p_i_minus1
                        # 对于反向延长，使用点 p_i_plus1 和 p_i_plus1 - (p_i_plus2 - p_i_plus1) = 2*p_i_plus1 - p_i_plus2
                        rev_point = (2 * p_i_plus1[0] - p_i_plus2[0], 2 * p_i_plus1[1] - p_i_plus2[1])
                        line2_p1 = p_i_plus1
                        line2_p2 = rev_point

                        intersection = line_intersection(line1_p1, line1_p2, line2_p1, line2_p2)
                        if intersection is None:
                            # 如果平行，添加中点作为fallback（根据问题假设应有交点，但安全处理）
                            intersection = ((line1_p1[0] + line2_p1[0]) / 2, (line1_p1[1] + line2_p1[1]) / 2)

                        points.insert(i, intersection)
                        points.pop((i + 1) % n)
                        points.pop(i-1)
                        n = len(points)

                else:

                    i += 1




        if len(points)==4:

            break
    points = invert_y_coordinates(points)
    return points  # 返回前4个点



def invert_y_coordinates(coordinates):


    return [(int(x), int(-y)) for x, y in coordinates]


def read_segments_coordinates(file_path, segment_names=None):

    segments = {}

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            seg_name = row['segment_name']

            # 如果指定了segment_names且当前segment不在其中，则跳过
            if segment_names is not None and seg_name not in segment_names:
                continue

            x = int(row['x'])
            y = int(row['y'])

            if seg_name not in segments:
                segments[seg_name] = []

            segments[seg_name].append((x, y))

    return segments


def write_segments_to_csv(output_path, segments_data, segment_names=None):

    max_points = max(len(segment) for segment in segments_data)
    header = ['segment_name']
    for i in range(1, max_points + 1):
        header.extend([f'x{i}', f'y{i}'])

    # 生成或使用提供的segment名称
    if segment_names is None:
        segment_names = [f'east_s{i + 1}' for i in range(len(segments_data))]
    elif len(segment_names) != len(segments_data):
        raise ValueError("segment_names长度必须与segments_data相同")

    # 写入文件
    with open(output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for name, segment in zip(segment_names, segments_data):
            row = [name]
            for point in segment:
                row.extend(point)
            writer.writerow(row)

def draw_heatmap_on_building(building_img, heatmap_img, x1, y1, x2, y2, x3, y3, x4, y4):
    building = cv2.imread(building_img, cv2.IMREAD_UNCHANGED)
    if building.shape[2] != 3:
        building = cv2.cvtColor(building, cv2.COLOR_BGRA2BGR)

    heatmap = cv2.imread(heatmap_img, cv2.IMREAD_UNCHANGED)
    if heatmap.shape[2] != 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGRA2BGR)



    # scale the heatmap to the original size
    heatmap_height, heatmap_width = heatmap.shape[:2]

    pts1 = np.float32([
        [0, 0], # top-left corner
        [heatmap_width, 0], # top-right corner
        [heatmap_width, heatmap_height], # bottom-right corner
        [0, heatmap_height] # bottom-left corner
    ])
    pts2 = np.float32([
        [x1, y1], # top-left corner
        [x2, y2], # top-right corner
        [x3, y3], # bottom-right corner
        [x4, y4] # bottom-left corner
    ])

    h, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    building_height, building_width = building.shape[:2]

    # warp the heatmap to the building's perspective
    warped = cv2.warpPerspective(heatmap, h, (building_width, building_height)) # warp info

    # Create a mask for the warped heatmap
    mask = np.zeros_like(building, dtype=np.uint8)

    # Fill the mask with the warped heatmap area
    cv2.fillConvexPoly(mask, pts2.astype(int), (255,255,255))

    # Ensure the mask is in the same format as the building image
    masked_dst = cv2.bitwise_and(building, cv2.bitwise_not(mask))

    # Combine the masked building with the warped heatmap
    output = cv2.add(masked_dst, warped)

    # save the result
    cv2.imwrite("./result.png", output)

    return "./result.png"

if __name__ == "__main__":
    test1 = read_segments_coordinates('./segments的副本.csv')
    print("输入:", test1, sep='\n')
    out = []
    for test in test1.values():
        result = process_coordinates(test)
        print("输出:", result)
        out.append(result)

    write_segments_to_csv('./output3.csv', out, test1.keys())

    building_img = "./input.jpg"  # Path to the building image

    segment = {}

    # read csv file to get segment and coordinates
    with open('./output.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader, None)
        for row in csv_reader:
            segment[row[0]] = list(map(int, row[1:]))

    output_img = building_img

    for key, value in segment.items():
        heatmap_img = f"./{key}.png"  # Path to the heatmap image
        x1, y1 = value[0], value[1]  # Top-left

        x2, y2 = value[2], value[3]  # Top-right
        x3, y3 = value[4], value[5]  # Bottom-right
        x4, y4 = value[6], value[7]  # Bottom-left
        
        output_img = draw_heatmap_on_building(output_img, heatmap_img, x1, y1, x2, y2, x3, y3, x4, y4)
    print("Heatmap drawn on building image and saved as 'result.png'.")
