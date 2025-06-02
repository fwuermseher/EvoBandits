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
    from sko.SA import SA_TSP

    from tsp import TSP  # Internal module
    from datasets import kro100C as dataset  # Internal module

    # Initialize the TSP
    n_cities = dataset.N_CITIES
    coordinates = dataset.COORDINATES
    best_route = dataset.BEST_TOUR
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    tsp = TSP(n_cities, dist_matrix)

    print(f"Initialized a TSP based on the {dataset.NAME} dataset")
    print(f"Known optimal tour:\n{best_route}")
    print(f"Known minimal distance:\t{tsp.calc_total_dist(best_route)}")

    # Run a Genetic Algorithm and visualize the results
    # Reference: https://scikit-opt.github.io/scikit-opt/#/en/README?id=_4-sasimulated-annealing
    algorithm = SA_TSP(
        func=tsp.calc_total_dist,
        x0=range(n_cities),
        T_max=100,
        T_min=1e-7,
        L=300,
        max_stay_counter=150,
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
        algorithm = SA_TSP(
            func=tsp.calc_total_dist,
            x0=range(n_cities),
            T_max=100,
            T_min=1e-7,
            L=300,
            max_stay_counter=150,
        )
        _, best_dist = algorithm.run()
        results.append(best_dist)
        print(f"Best distance (sample #{i}):\t{best_dist}")

    print(f"Mean of Sample Results:\t{np.mean(results)}")
    print(f"Standard deviation:\t{np.sqrt(np.var(results))}")


#  Initialized a TSP based on the kro100C dataset.
# Known optimal tour:
# [ 0 84 26 14 12 78 63 19 41 54 66 46 30 64 79 76 29 67 34  1 53  5 74 21
#   7 16 24 89 33 57 97 87 27 38 37 70 55 42  4 85 71 82 61 49 94 93 90 75
#  69 22 20 88 40 58 72  2 68 59  3 92 98 18 91  9 13 35 56 73 99 32 44 80
#  96 95 86 51 10 83 47 65 43 62 50 15 36  8 77 81  6 25 60 31 23 45 28 17
#  48 11 39 52]
# Known minimal distance: 20750.762503687536
# Example Results obtained from Simulated Annealing:
# Number of Samples:      100
# Best distance (sample #0):      22249.487003100538
# Best distance (sample #1):      22109.295334896677
# Best distance (sample #2):      23815.045562366908
# Best distance (sample #3):      21320.956144154326
# Best distance (sample #4):      22120.524975955974
# Best distance (sample #5):      22096.95544789991
# Best distance (sample #6):      21706.93048379798
# Best distance (sample #7):      21133.31284808919
# Best distance (sample #8):      21650.1245035373
# Best distance (sample #9):      22671.448573955277
# Best distance (sample #10):     21730.291096308472
# Best distance (sample #11):     21280.648129885456
# Best distance (sample #12):     21793.10825054839
# Best distance (sample #13):     21280.124005647616
# Best distance (sample #14):     22098.24972052906
# Best distance (sample #15):     23201.36748110768
# Best distance (sample #16):     22204.234252593076
# Best distance (sample #17):     21164.94558050773
# Best distance (sample #18):     21369.13342772894
# Best distance (sample #19):     21499.64045894204
# Best distance (sample #20):     23586.633145610416
# Best distance (sample #21):     22151.163304317295
# Best distance (sample #22):     22185.981870406657
# Best distance (sample #23):     22165.46588686295
# Best distance (sample #24):     21799.964632328
# Best distance (sample #25):     21503.793548905916
# Best distance (sample #26):     22555.875943824398
# Best distance (sample #27):     22338.326471237273
# Best distance (sample #28):     22344.77319643373
# Best distance (sample #29):     22135.05756879895
# Best distance (sample #30):     22886.99310023957
# Best distance (sample #31):     23126.5284344892
# Best distance (sample #32):     22036.63802874384
# Best distance (sample #33):     21648.163104195763
# Best distance (sample #34):     22183.924100180877
# Best distance (sample #35):     23114.224760683723
# Best distance (sample #36):     22428.823735575304
# Best distance (sample #37):     22090.16379608852
# Best distance (sample #38):     21103.204328193784
# Best distance (sample #39):     23258.67747761154
# Best distance (sample #40):     21235.03816590651
# Best distance (sample #41):     21207.797459336947
# Best distance (sample #42):     22634.41039657501
# Best distance (sample #43):     23026.947445797538
# Best distance (sample #44):     22551.66383934416
# Best distance (sample #45):     21942.060513329074
# Best distance (sample #46):     21459.52821538908
# Best distance (sample #47):     21141.856063870215
# Best distance (sample #48):     22288.85081285318
# Best distance (sample #49):     20988.372387519277
# Best distance (sample #50):     21408.003399269077
# Best distance (sample #51):     21742.996967497165
# Best distance (sample #52):     22333.771832263916
# Best distance (sample #53):     22281.201537795125
# Best distance (sample #54):     21562.798444031316
# Best distance (sample #55):     21103.84539566153
# Best distance (sample #56):     21502.472051958637
# Best distance (sample #57):     22635.631363451706
# Best distance (sample #58):     21548.45926742559
# Best distance (sample #59):     21464.32829458145
# Best distance (sample #60):     22282.11312023191
# Best distance (sample #61):     23464.694533197693
# Best distance (sample #62):     22193.405996292622
# Best distance (sample #63):     22487.645619738018
# Best distance (sample #64):     21454.78726008508
# Best distance (sample #65):     21886.517149516214
# Best distance (sample #66):     22519.92268813758
# Best distance (sample #67):     22534.49664283957
# Best distance (sample #68):     22709.20630280357
# Best distance (sample #69):     21773.142641803148
# Best distance (sample #70):     23400.229377259206
# Best distance (sample #71):     20962.422499198638
# Best distance (sample #72):     22543.42443815615
# Best distance (sample #73):     22764.500590871357
# Best distance (sample #74):     22448.747541743654
# Best distance (sample #75):     22071.164662842755
# Best distance (sample #76):     22951.379128503435
# Best distance (sample #77):     23124.385562003437
# Best distance (sample #78):     22104.947672816208
# Best distance (sample #79):     21824.930595759786
# Best distance (sample #80):     21698.18510920759
# Best distance (sample #81):     22628.2764238536
# Best distance (sample #82):     21230.47127825406
# Best distance (sample #83):     23025.93507541405
# Best distance (sample #84):     21690.938418977526
# Best distance (sample #85):     21723.703177903226
# Best distance (sample #86):     22167.707228318995
# Best distance (sample #87):     22098.58024123968
# Best distance (sample #88):     22104.715275761962
# Best distance (sample #89):     21224.782890051425
# Best distance (sample #90):     21532.170007209857
# Best distance (sample #91):     22093.974959974734
# Best distance (sample #92):     21377.26499466149
# Best distance (sample #93):     21666.538832556616
# Best distance (sample #94):     21576.109606834787
# Best distance (sample #95):     23304.014674228347
# Best distance (sample #96):     22521.491445320316
# Best distance (sample #97):     22646.690311662955
# Best distance (sample #98):     21426.008901688278
# Best distance (sample #99):     21542.146089915983
# Mean of Sample Results: 22079.82002536972
# Standard deviation:     655.7749673958648
