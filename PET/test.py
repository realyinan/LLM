from rich import print


def get_common_sub_str(str1: str, str2: str):
    len_str1= len(str1)
    len_str2 = len(str2)
    # 定义一个数组，每一个元素代表最大公共子串的长度
    record = [[0 for _ in range(len_str2+1)] for _ in range(len_str1+1)]
    max_len = 0
    end_idx = 0
    for i in range(1, len_str1+1):
        for j in range(1, len_str2+1):
            if str1[i-1] == str2[j-1]:
                record[i][j] = record[i-1][j-1] + 1
            if max_len < record[i][j]:
                max_len = record[i][j]
                end_idx = i
    return str1[end_idx-max_len: end_idx], max_len



if __name__ == '__main__':
    str1 = "abcd"
    str2 = "abadbcdba"
    print(get_common_sub_str(str1, str2))