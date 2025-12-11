import numpy as np
from sortedcontainers import SortedList


def decode(n, J, P, s):
    job_num = len(J)
    process_num = 0
    max_process_num = 0
    machine_list = []

    for i in range(job_num):
        current_pro_num = len(J[i])
        machine_list_by_process = s[process_num:process_num + current_pro_num]
        max_process_num = max(max_process_num, current_pro_num)
        process_num += current_pro_num
        machine_list.append(machine_list_by_process)

    s = s[process_num:]

    # 初始化T和C矩阵
    T = [SortedList() for _ in range(n)]
    C = np.zeros((job_num, max_process_num), dtype=int)
    k = np.zeros(job_num, dtype=int)

    # 遍历任务序列进行处理
    for job in s:
        machine_index = machine_list[job][k[job]]
        machine = J[job][k[job]][machine_index]
        process_time = P[job][k[job]][machine_index]
        last_job_finish = C[job, k[job] - 1] if k[job] > 0 else 0
        start_time = max(last_job_finish, T[machine][-1][-1] if T[machine] else 0)
        insert_index = T[machine].bisect_left([start_time, 0, 0, 0])
        end_time = start_time + process_time
        C[job, k[job]] = end_time
        T[machine].add([start_time, job, k[job], end_time])
        k[job] += 1
    C = np.array([np.insert(complete, 0, i) for i, complete in enumerate(C)])
    return T, C


def main():
    n = 3
    J = [
        [[0, 2], [1, 2]],
        [[0, 1], [0, 1, 2], [0, 2]],
        [[1, 2], [0, 1]],
        [[1, 2], [0, 1, 2], [0, 1]]
    ]
    P = [
        [[6, 5], [7, 4]],
        [[8, 6], [9, 7, 6], [5, 4]],
        [[5, 6], [7, 4]],
        [[8, 6], [4, 5, 3], [6, 7]]
    ]
    s = [1, 0, 1, 2, 0, 0, 0, 1, 0, 1, 3, 1, 3, 2, 0, 3, 0, 1, 2, 1]
    T, C = decode(n, J, P, s)

    print("===== Machine Schedule (T) =====")
    for i, machine in enumerate(T):
        print(f"Machine {i}:")
        for seg in machine:
            print(f"  start={seg[0]}, job={seg[1]}, op={seg[2]}, end={seg[3]}")

    print("\n===== Completion Matrix (C) =====")
    print(C)

if __name__ == "__main__":
    main()