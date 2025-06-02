# Copyright 2025 EvoBandits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if __name__ == "__main__":
    """Runs a Genetic Algorithm to solve a Traveling Salesman Problem."""

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from sko.GA import GA_TSP

    from tsp import TSP  # Internal module

    # TSP based on the kroC dataset, available under:
    # https://github.com/mahf-opt/mahf-tsplib/blob/master/src/tsplib/kroc100.tsp
    n_cities = 100
    # Coordinates of 100 cities (from kroC dataset)
    coordinates = np.array([
        [1357, 1905], [2650, 802], [1774, 107], [1307, 964], [3806, 746],
        [2687, 1353], [43, 1957], [3092, 1668], [185, 1542], [834, 629],
        [40, 462], [1183, 1391], [2048, 1628], [1097, 643], [1838, 1732],
        [234, 1118], [3314, 1881], [737, 1285], [779, 777], [2312, 1949],
        [2576, 189], [3078, 1541], [2781, 478], [705, 1812], [3409, 1917],
        [323, 1714], [1660, 1556], [3729, 1188], [693, 1383], [2361, 640],
        [2433, 1538], [554, 1825], [913, 317], [3586, 1909], [2636, 727],
        [1000, 457], [482, 1337], [3704, 1082], [3635, 1174], [1362, 1526],
        [2049, 417], [2552, 1909], [3939, 640], [219, 898], [812, 351],
        [901, 1552], [2513, 1572], [242, 584], [826, 1226], [3278, 799],
        [86, 1065], [14, 454], [1327, 1893], [2773, 1286], [2469, 1838],
        [3835, 963], [1031, 428], [3853, 1712], [1868, 197], [1544, 863],
        [457, 1607], [3174, 1064], [192, 1004], [2318, 1925], [2232, 1374],
        [396, 828], [2365, 1649], [2499, 658], [1410, 307], [2990, 214],
        [3646, 1018], [3394, 1028], [1779, 90], [1058, 372], [2933, 1459],
        [3099, 173], [2178, 978], [138, 1610], [2082, 1753], [2302, 1127],
        [805, 272], [22, 1617], [3213, 1085], [99, 536], [1533, 1780],
        [3564, 676], [29, 6], [3808, 1375], [2221, 291], [3499, 1885],
        [3124, 408], [781, 671], [1027, 1041], [3249, 378], [3297, 491],
        [213, 220], [721, 186], [3736, 1542], [868, 731], [960, 303],
    ])

    # Known optimal tour (0-based indices)
    best_route = np.array([
        0, 84, 26, 14, 12, 78, 63, 19, 41, 54, 66, 46, 30, 64, 79, 76, 29, 67, 34, 1,
        53, 5, 74, 21, 7, 16, 24, 89, 33, 57, 97, 87, 27, 38, 37, 70, 55, 42, 4, 85,
        71, 82, 61, 49, 94, 93, 90, 75, 69, 22, 20, 88, 40, 58, 72, 2, 68, 59, 3, 92,
        98, 18, 91, 9, 13, 35, 56, 73, 99, 32, 44, 80, 96, 95, 86, 51, 10, 83, 47, 65,
        43, 62, 50, 15, 36, 8, 77, 81, 6, 25, 60, 31, 23, 45, 28, 17, 48, 11, 39, 52
    ])
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    tsp = TSP(n_cities, dist_matrix)

    print("Initialized a TSP based on the kroC dataset with 100 cities.")
    print(f"Known optimal tour:\n{best_route}")
    print(f"Known minimal distance:\t{tsp.calc_total_dist(best_route)}")

    # Run a Genetic Algorithm and visualize the results
    # based on: https://scikit-opt.github.io/scikit-opt/#/en/README?id=_22-genetic-algorithm-for-tsptravelling-salesman-problem
    algorithm = GA_TSP(
        func=tsp.calc_total_dist,
        n_dim=n_cities,
        size_pop=50,
        max_iter=2500,
        prob_mut=0.9,
    )
    best_points, best_distance = algorithm.run()
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(coordinates[:, 0], coordinates[:, 1], "o-r")
    ax[1].plot(algorithm.generation_best_Y)
    plt.show()

    # Assess noise of the objective
    n_samples = 100
    results = []
    print("Example Results obtained from a Genetic Algorithm:")
    print(f"Number of Samples:\t{n_samples}")

    for i in range(n_samples):
        algorithm = GA_TSP(
            func=tsp.calc_total_dist,
            n_dim=n_cities,
            size_pop=50,
            max_iter=2500,
            prob_mut=0.9,
        )
        _, best_dist = algorithm.run()
        results.append(best_dist)
        print(f"Best distance (sample #{i}):\t{best_dist}")

    print(f"Mean of Sample Results:\t{np.mean(results)}")
    print(f"Standard deviation:\t{np.sqrt(np.var(results))}")


# Initialized a TSP based on the kroC dataset with 100 cities.
# Known optimal tour:
# [ 0 84 26 14 12 78 63 19 41 54 66 46 30 64 79 76 29 67 34  1 53  5 74 21
#   7 16 24 89 33 57 97 87 27 38 37 70 55 42  4 85 71 82 61 49 94 93 90 75
#  69 22 20 88 40 58 72  2 68 59  3 92 98 18 91  9 13 35 56 73 99 32 44 80
#  96 95 86 51 10 83 47 65 43 62 50 15 36  8 77 81  6 25 60 31 23 45 28 17
#  48 11 39 52]
# Known minimal distance: 20750.762503687536
# Example Results obtained from a Genetic Algorithm:
# Number of Samples:      100
# Best distance (sample #0):      [24198.70219243]
# Best distance (sample #1):      [22511.66636736]
# Best distance (sample #2):      [24004.09571543]
# Best distance (sample #3):      [22960.64323373]
# Best distance (sample #4):      [23624.75091715]
# Best distance (sample #5):      [21222.35848267]
# Best distance (sample #6):      [22845.96331101]
# Best distance (sample #7):      [22033.99136412]
# Best distance (sample #8):      [22066.79160599]
# Best distance (sample #9):      [21784.01417776]
# Best distance (sample #10):     [23566.74829253]
# Best distance (sample #11):     [22378.07112273]
# Best distance (sample #12):     [21107.09393613]
# Best distance (sample #13):     [23101.25309927]
# Best distance (sample #14):     [22159.06739562]
# Best distance (sample #15):     [22916.66007809]
# Best distance (sample #16):     [22878.22908305]
# Best distance (sample #17):     [22392.02219903]
# Best distance (sample #18):     [23401.75949729]
# Best distance (sample #19):     [23028.75243921]
# Best distance (sample #20):     [22274.53812376]
# Best distance (sample #21):     [22140.92735011]
# Best distance (sample #22):     [22164.42573233]
# Best distance (sample #23):     [22728.71822148]
# Best distance (sample #24):     [22464.84029845]
# Best distance (sample #25):     [22910.26491765]
# Best distance (sample #26):     [23136.46633379]
# Best distance (sample #27):     [23561.30808659]
# Best distance (sample #28):     [21898.59894755]
# Best distance (sample #29):     [21970.70811347]
# Best distance (sample #30):     [23589.21982385]
# Best distance (sample #31):     [22811.03882266]
# Best distance (sample #32):     [21376.82260923]
# Best distance (sample #33):     [22628.65158411]
# Best distance (sample #34):     [21900.60135977]
# Best distance (sample #35):     [21986.88655663]
# Best distance (sample #36):     [23421.22917949]
# Best distance (sample #37):     [22256.24146134]
# Best distance (sample #38):     [23503.60519072]
# Best distance (sample #39):     [21754.73371627]
# Best distance (sample #40):     [22086.38536346]
# Best distance (sample #41):     [22160.75625667]
# Best distance (sample #42):     [22847.58151001]
# Best distance (sample #43):     [23103.65538731]
# Best distance (sample #44):     [22386.28630277]
# Best distance (sample #45):     [23338.68005845]
# Best distance (sample #46):     [22104.35572109]
# Best distance (sample #47):     [21603.89005699]
# Best distance (sample #48):     [22144.21521114]
# Best distance (sample #49):     [22162.69882812]
# Best distance (sample #50):     [24161.92759382]
# Best distance (sample #51):     [23025.14998824]
# Best distance (sample #52):     [22913.34123479]
# Best distance (sample #53):     [21020.72744777]
# Best distance (sample #54):     [22863.22114847]
# Best distance (sample #55):     [21581.53865505]
# Best distance (sample #56):     [21252.85849044]
# Best distance (sample #57):     [21574.6183178]
# Best distance (sample #58):     [21721.86584944]
# Best distance (sample #59):     [22196.67289001]
# Best distance (sample #60):     [23305.38558137]
# Best distance (sample #61):     [22966.99175066]
# Best distance (sample #62):     [22389.54767337]
# Best distance (sample #63):     [22967.82924632]
# Best distance (sample #64):     [23171.18304714]
# Best distance (sample #65):     [22969.51280487]
# Best distance (sample #66):     [22600.60693402]
# Best distance (sample #67):     [21948.55471071]
# Best distance (sample #68):     [22740.2827373]
# Best distance (sample #69):     [23445.12168679]
# Best distance (sample #70):     [22863.37139129]
# Best distance (sample #71):     [22491.13956329]
# Best distance (sample #72):     [21646.8192922]
# Best distance (sample #73):     [23375.32345288]
# Best distance (sample #74):     [22390.27005001]
# Best distance (sample #75):     [22814.71007652]
# Best distance (sample #76):     [21960.34192167]
# Best distance (sample #77):     [21609.66071794]
# Best distance (sample #78):     [24265.71874722]
# Best distance (sample #79):     [22810.7470283]
# Best distance (sample #80):     [21580.63714753]
# Best distance (sample #81):     [22521.42022871]
# Best distance (sample #82):     [22108.79236465]
# Best distance (sample #83):     [22044.56128362]
# Best distance (sample #84):     [23745.67400673]
# Best distance (sample #85):     [22348.92651141]
# Best distance (sample #86):     [22287.93583471]
# Best distance (sample #87):     [22444.50666611]
# Best distance (sample #88):     [22788.10358495]
# Best distance (sample #89):     [21481.8105555]
# Best distance (sample #90):     [22168.22795696]
# Best distance (sample #91):     [22516.91739415]
# Best distance (sample #92):     [23376.27040671]
# Best distance (sample #93):     [23308.82860203]
# Best distance (sample #94):     [22714.11628385]
# Best distance (sample #95):     [22774.87081359]
# Best distance (sample #96):     [22809.96714515]
# Best distance (sample #97):     [23288.43790794]
# Best distance (sample #98):     [23459.89898339]
# Best distance (sample #99):     [22686.45473858]
# Mean of Sample Results: 22581.0236408182
# Standard deviation:     710.5421287793392
