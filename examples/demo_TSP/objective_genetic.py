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
    # based on: https://scikit-opt.github.io/scikit-opt/#/en/README?id=_22-genetic-algorithm-for-tsptravelling-salesman-problem
    algorithm = GA_TSP(
        func=tsp.calc_total_dist,
        n_dim=n_cities,
        size_pop=20,
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

#
###################### KRO100C #####################
#
# Initialized a TSP based on the kro100C dataset.
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
#
##################### LIN318 #####################
#
# Initialized a TSP based on the lin318 dataset
# Known optimal tour:
# [0 0]
# Known minimal distance: 0.0
# Example Results obtained from a Genetic Algorithm:
# Number of Samples:      100
# Best distance (sample #0):      [114311.88128055]
# Best distance (sample #1):      [114562.88270243]
# Best distance (sample #2):      [125533.09786675]
# Best distance (sample #3):      [128723.22521214]
# Best distance (sample #4):      [118592.50945522]
# Best distance (sample #5):      [126428.15322834]
# Best distance (sample #6):      [125418.72563832]
# Best distance (sample #7):      [120423.99101675]
# Best distance (sample #8):      [122884.35799207]
# Best distance (sample #9):      [113062.37736291]
# Best distance (sample #10):     [114153.68852318]
# Best distance (sample #11):     [119071.7569629]
# Best distance (sample #12):     [118879.39059878]
# Best distance (sample #13):     [122273.72898874]
# Best distance (sample #14):     [116255.55174613]
# Best distance (sample #15):     [118200.88057801]
# Best distance (sample #16):     [124727.4972841]
# Best distance (sample #17):     [118361.43701285]
# Best distance (sample #18):     [123827.44520929]
# Best distance (sample #19):     [111174.26920009]
# Best distance (sample #20):     [118733.63825367]
# Best distance (sample #21):     [129163.63961267]
# Best distance (sample #22):     [126051.97543935]
# Best distance (sample #23):     [118789.86885777]
# Best distance (sample #24):     [120731.22875334]
# Best distance (sample #25):     [120163.85947675]
# Best distance (sample #26):     [119662.52292471]
# Best distance (sample #27):     [120736.86809164]
# Best distance (sample #28):     [123103.96612881]
# Best distance (sample #29):     [113721.93093944]
# Best distance (sample #30):     [124065.6834928]
# Best distance (sample #31):     [122933.44167766]
# Best distance (sample #32):     [117638.26930244]
# Best distance (sample #33):     [118031.19822447]
# Best distance (sample #34):     [121814.45081451]
# Best distance (sample #35):     [119323.76213751]
# Best distance (sample #36):     [119244.90229364]
# Best distance (sample #37):     [124405.73087708]
# Best distance (sample #38):     [116894.36820472]
# Best distance (sample #39):     [119515.76882205]
# Best distance (sample #40):     [122560.5399742]
# Best distance (sample #41):     [113148.2540884]
# Best distance (sample #42):     [122080.89350084]
# Best distance (sample #43):     [119838.86527207]
# Best distance (sample #44):     [110279.20075531]
# Best distance (sample #45):     [121164.5291874]
# Best distance (sample #46):     [107305.3291376]
# Best distance (sample #47):     [120469.832569]
# Best distance (sample #48):     [117984.08637167]
# Best distance (sample #49):     [124120.39971814]
# Best distance (sample #50):     [118310.24770853]
# Best distance (sample #51):     [124385.27894974]
# Best distance (sample #52):     [117616.43805567]
# Best distance (sample #53):     [120984.68067555]
# Best distance (sample #54):     [125339.62962203]
# Best distance (sample #55):     [108628.30009274]
# Best distance (sample #56):     [111091.58429623]
# Best distance (sample #57):     [110823.89940075]
# Best distance (sample #58):     [120611.98311144]
# Best distance (sample #59):     [114598.42094094]
# Best distance (sample #60):     [118800.92685478]
# Best distance (sample #61):     [124552.03364317]
# Best distance (sample #62):     [107693.49852901]
# Best distance (sample #63):     [119001.3507915]
# Best distance (sample #64):     [118387.51442467]
# Best distance (sample #65):     [116064.60851755]
# Best distance (sample #66):     [116232.22840067]
# Best distance (sample #67):     [118946.38410011]
# Best distance (sample #68):     [115908.89955349]
# Best distance (sample #69):     [119036.91574677]
# Best distance (sample #70):     [129154.59451227]
# Best distance (sample #71):     [104640.4743291]
# Best distance (sample #72):     [117256.85493187]
# Best distance (sample #73):     [125362.80600073]
# Best distance (sample #74):     [117315.57859302]
# Best distance (sample #75):     [125837.22538056]
# Best distance (sample #76):     [116777.67358339]
# Best distance (sample #77):     [119617.66120341]
# Best distance (sample #78):     [123729.71299363]
# Best distance (sample #79):     [127675.55402515]
# Best distance (sample #80):     [119161.34557734]
# Best distance (sample #81):     [113780.15127061]
# Best distance (sample #82):     [123970.32036667]
# Best distance (sample #83):     [117600.71365515]
# Best distance (sample #84):     [119247.26752463]
# Best distance (sample #85):     [119609.40015422]
# Best distance (sample #86):     [109505.61898937]
# Best distance (sample #87):     [116457.22202918]
# Best distance (sample #88):     [110600.12240208]
# Best distance (sample #89):     [117423.29204524]
# Best distance (sample #90):     [121561.38645866]
# Best distance (sample #91):     [120266.3245211]
# Best distance (sample #92):     [123055.29379254]
# Best distance (sample #93):     [124122.70197508]
# Best distance (sample #94):     [120640.11241677]
# Best distance (sample #95):     [120664.79178944]
# Best distance (sample #96):     [117714.50117525]
# Best distance (sample #97):     [118756.85810277]
# Best distance (sample #98):     [128660.12166997]
# Best distance (sample #99):     [124795.1318085]
# Mean of Sample Results: 119325.21489452252
# Standard deviation:     5006.685157978833
#
##################### KRO100C + NOISE ############
#
# Initialized a TSP based on the kro100C dataset
# Known optimal tour:
# [ 0 84 26 14 12 78 63 19 41 54 66 46 30 64 79 76 29 67 34  1 53  5 74 21
#   7 16 24 89 33 57 97 87 27 38 37 70 55 42  4 85 71 82 61 49 94 93 90 75
#  69 22 20 88 40 58 72  2 68 59  3 92 98 18 91  9 13 35 56 73 99 32 44 80
#  96 95 86 51 10 83 47 65 43 62 50 15 36  8 77 81  6 25 60 31 23 45 28 17
#  48 11 39 52]
# Known minimal distance: 19554.810452390826
# Example Results obtained from a Genetic Algorithm:
# Number of Samples:      100
# Best distance (sample #0):      [23827.7135989]
# Best distance (sample #1):      [24167.43905337]
# Best distance (sample #2):      [24300.52468917]
# Best distance (sample #3):      [26057.47160503]
# Best distance (sample #4):      [22753.73964257]
# Best distance (sample #5):      [28009.59736719]
# Best distance (sample #6):      [25160.63682072]
# Best distance (sample #7):      [26550.19882775]
# Best distance (sample #8):      [26187.40326388]
# Best distance (sample #9):      [27282.92170656]
# Best distance (sample #10):     [27593.50445212]
# Best distance (sample #11):     [25711.33174405]
# Best distance (sample #12):     [27416.55483406]
# Best distance (sample #13):     [25253.76631807]
# Best distance (sample #14):     [25102.44107143]
# Best distance (sample #15):     [25211.9791742]
# Best distance (sample #16):     [25380.01852349]
# Best distance (sample #17):     [22171.79300814]
# Best distance (sample #18):     [26477.92338514]
# Best distance (sample #19):     [27289.94118579]
# Best distance (sample #20):     [25685.02593692]
# Best distance (sample #21):     [25352.7211493]
# Best distance (sample #22):     [25564.59084843]
# Best distance (sample #23):     [25918.43370172]
# Best distance (sample #24):     [27263.9346492]
# Best distance (sample #25):     [25580.33404284]
# Best distance (sample #26):     [23988.31016177]
# Best distance (sample #27):     [24612.52841605]
# Best distance (sample #28):     [25973.87174081]
# Best distance (sample #29):     [24403.54514234]
# Best distance (sample #30):     [27630.89530858]
# Best distance (sample #31):     [26799.78915291]
# Best distance (sample #32):     [23903.9025566]
# Best distance (sample #33):     [24488.12093341]
# Best distance (sample #34):     [27497.13138543]
# Best distance (sample #35):     [24518.55698655]
# Best distance (sample #36):     [26432.75576391]
# Best distance (sample #37):     [25058.91324802]
# Best distance (sample #38):     [23943.33919328]
# Best distance (sample #39):     [25372.50793847]
# Best distance (sample #40):     [25166.53056156]
# Best distance (sample #41):     [24626.70012577]
# Best distance (sample #42):     [24516.57224868]
# Best distance (sample #43):     [26411.81845668]
# Best distance (sample #44):     [26298.46823529]
# Best distance (sample #45):     [25737.43204706]
# Best distance (sample #46):     [23058.47792355]
# Best distance (sample #47):     [25726.40903949]
# Best distance (sample #48):     [26818.07869603]
# Best distance (sample #49):     [24788.74295928]
# Best distance (sample #50):     [23553.16069867]
# Best distance (sample #51):     [25649.3938607]
# Best distance (sample #52):     [25558.54935208]
# Best distance (sample #53):     [26374.12643628]
# Best distance (sample #54):     [25362.28638598]
# Best distance (sample #55):     [23827.58647766]
# Best distance (sample #56):     [27194.14913428]
# Best distance (sample #57):     [27774.52218001]
# Best distance (sample #58):     [26307.84826336]
# Best distance (sample #59):     [24819.56069038]
# Best distance (sample #60):     [30108.98400304]
# Best distance (sample #61):     [25635.95767689]
# Best distance (sample #62):     [25306.28646799]
# Best distance (sample #63):     [24051.82480049]
# Best distance (sample #64):     [25615.85014886]
# Best distance (sample #65):     [25014.37141292]
# Best distance (sample #66):     [27478.69601097]
# Best distance (sample #67):     [24535.65366495]
# Best distance (sample #68):     [24965.15272591]
# Best distance (sample #69):     [24089.99206868]
# Best distance (sample #70):     [25747.04916]
# Best distance (sample #71):     [26507.59841173]
# Best distance (sample #72):     [26044.41981161]
# Best distance (sample #73):     [25525.82903965]
# Best distance (sample #74):     [24052.45781057]
# Best distance (sample #75):     [28583.93200675]
# Best distance (sample #76):     [24621.19586221]
# Best distance (sample #77):     [25390.72837038]
# Best distance (sample #78):     [24164.96591235]
# Best distance (sample #79):     [24124.54815756]
# Best distance (sample #80):     [24576.13457704]
# Best distance (sample #81):     [25164.89445598]
# Best distance (sample #82):     [25746.24739264]
# Best distance (sample #83):     [23793.18276237]
# Best distance (sample #84):     [25274.00248769]
# Best distance (sample #85):     [26095.54200388]
# Best distance (sample #86):     [27255.68368606]
# Best distance (sample #87):     [26618.20062796]
# Best distance (sample #88):     [28536.87454321]
# Best distance (sample #89):     [24766.88216888]
# Best distance (sample #90):     [25892.56975714]
# Best distance (sample #91):     [26037.84843104]
# Best distance (sample #92):     [23419.66918808]
# Best distance (sample #93):     [23024.80076553]
# Best distance (sample #94):     [26959.09271112]
# Best distance (sample #95):     [26853.23994134]
# Best distance (sample #96):     [25203.70098817]
# Best distance (sample #97):     [24574.05042622]
# Best distance (sample #98):     [25522.71693675]
# Best distance (sample #99):     [25848.98545519]
# Mean of Sample Results: 25541.963351307713
# Standard deviation:     1361.477800530837
