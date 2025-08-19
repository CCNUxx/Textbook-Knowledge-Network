import numpy as np
import pandas as pd
import itertools

excel_file = "data/Knowledge Point information.xlsx"

knowledge_points = pd.read_excel(excel_file, usecols=[0])
knowledge_points = knowledge_points.iloc[:, 0].tolist()

knowledge_points.sort(key=len)
knowledge_points = knowledge_points[::-1]

mat = np.zeros((len(knowledge_points), len(knowledge_points)))

txt_file = []

# Open a file
version = "test"
with open("data/test.txt", "r", encoding="utf-8") as file:
    line = file.readline()
    while line:
        txt_file.append(line)
        # print(line, end="")
        line = file.readline()

# Paragraph Integration
data = []
i = 0
while True:
    if i >= len(txt_file):
        break
    txt_paragraph = txt_file[i].replace(" ", "")  # Remove spaces
    if txt_paragraph == "\n":
        i += 1
    else:
        j = 1
        while True:
            if i + j >= len(txt_file):
                data.append(txt_paragraph)
                i = len(txt_file)
                break
            txt_tmp = txt_file[i + j]
            if txt_tmp == "\n":
                data.append(txt_paragraph)
                break
            else:
                txt_paragraph = txt_paragraph.split("\n")[0] + txt_tmp.replace(" ", "")
            j += 1
        i += j

print(f"Total number of paragraphs: {len(data)}")

# Connection relationship
connect = []
all_used_knowledge_points = set()
for i in range(len(data)):
    tmp_paragraph = data[i]

    if "$$$" in tmp_paragraph or "&&&" in tmp_paragraph:
        continue

    count = 0
    tmp_knowledge_points = []
    for j in range(len(knowledge_points)):
        if knowledge_points[j] in tmp_paragraph:
            tmp_knowledge_points.append(knowledge_points[j])
            all_used_knowledge_points.add(knowledge_points[j])
            tmp_paragraph = tmp_paragraph.replace(knowledge_points[j], "")
            count += 1
    tmp_knowledge_points = list(set(tmp_knowledge_points))

    if count >= 2:
        connect.append(tmp_knowledge_points)
        # print(i + 1, tmp_knowledge_points)

# Constructing the matrix
filtered_knowledge_points = [kp for kp in knowledge_points if kp in all_used_knowledge_points]
mat = np.zeros((len(filtered_knowledge_points), len(filtered_knowledge_points)))

for i in range(len(connect)):
    tmp_connect = connect[i]
    result = list(itertools.permutations(tmp_connect, 2))
    for j in range(len(result)):
        word_1 = result[j][0]
        word_2 = result[j][1]

        if word_1 in filtered_knowledge_points and word_2 in filtered_knowledge_points:
            idx_1 = filtered_knowledge_points.index(word_1)
            idx_2 = filtered_knowledge_points.index(word_2)
            mat[idx_1, idx_2] += 1

for kp in all_used_knowledge_points:
    if kp not in filtered_knowledge_points:
        filtered_knowledge_points.append(kp)

# 生成 DataFrame
result = pd.DataFrame(
    mat, index=filtered_knowledge_points, columns=filtered_knowledge_points
)

result.to_excel(f"Result/{version}-matrix.xlsx")

print("end")


