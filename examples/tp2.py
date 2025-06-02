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

import heapq
import math
import numpy as np


SEED = 42
RNG = np.random.default_rng(SEED)


class PQElement:
    def __init__(self, customer_index, demand, product_index, timestep):
        self.customer_index = customer_index
        self.demand = demand
        self.product_index = product_index
        self.timestep = timestep

    def __lt__(self, other):
        return self.timestep < other.timestep


def function_value(action_vector: list[int], minimize=False, noise=True):
    PQ = []
    lambda_vals = [3.6, 3.0, 2.4, 1.8, 1.2]
    inventory = [action_vector[i] for i in range(8)]
    profit_per_item = [1, 2, 3, 4, 5, 6, 7, 8]
    holding_cost_per_item = [2] * 8
    mu = [0.15, 0.40, 0.25, 0.15, 0.25, 0.08, 0.13, 0.40]
    sigma = [0.0225, 0.0600, 0.0375, 0.0225, 0.0375, 0.0120, 0.0195, 0.0600]

    for i, lam in enumerate(lambda_vals):
        t = 0.0
        while t <= 70.0:
            x = RNG.uniform(0, 1)
            rv = -math.log(1 - x) / lam
            t += rv
            if 0.0 < t <= 70.0:
                heapq.heappush(PQ, PQElement(i + 1, 1, 0, t))

    heapq.heappush(PQ, PQElement(0, 1, 0, 20.0))
    heapq.heappush(PQ, PQElement(0, 1, 0, 70.0))

    profit = 0.0
    last_timestep = 0.0

    while PQ:
        current_event = heapq.heappop(PQ)
        delta_t = current_event.timestep - last_timestep
        last_timestep = current_event.timestep

        sold = [0] * 8
        customer_served = False

        if current_event.demand:
            if current_event.customer_index == 1:
                if inventory[0] > 0 and inventory[3] > 0 and inventory[5] > 0:
                    sold[0] = sold[3] = sold[5] = 1
                    customer_served = True
                if inventory[6] > 0:
                    sold[6] = 1
            elif current_event.customer_index == 2:
                if inventory[0] > 0 and inventory[4] > 0 and inventory[5] > 0:
                    sold[0] = sold[4] = sold[5] = 1
                    customer_served = True
                if inventory[6] > 0:
                    sold[6] = 1
            elif current_event.customer_index == 3:
                if inventory[1] > 0 and inventory[3] > 0 and inventory[5] > 0:
                    sold[1] = sold[3] = sold[5] = 1
                    customer_served = True
            elif current_event.customer_index == 4:
                if inventory[2] > 0 and inventory[3] > 0 and inventory[5] > 0:
                    sold[2] = sold[3] = sold[5] = 1
                    customer_served = True
                if inventory[7] > 0:
                    sold[7] = 1
            elif current_event.customer_index == 5:
                if inventory[2] > 0 and inventory[4] > 0 and inventory[5] > 0:
                    sold[2] = sold[4] = sold[5] = 1
                    customer_served = True
                if inventory[6] > 0:
                    sold[6] = 1

        for i in range(len(inventory)):
            profit -= inventory[i] * holding_cost_per_item[i] * delta_t

        if customer_served:
            for i in range(len(inventory)):
                inventory[i] -= sold[i]
                profit += sold[i] * profit_per_item[i]
                if sold[i] == 1:
                    rv = RNG.normal(mu[i], sigma[i])
                    if rv < 0:
                        print("error ZV <0")
                    heapq.heappush(PQ, PQElement(0, 0, i, rv + current_event.timestep))

        if not current_event.demand:
            for i in range(len(inventory)):
                if current_event.product_index == i:
                    inventory[i] += 1

        if current_event.timestep == 20:
            profit = 0.0
        if current_event.timestep == 70.0:
            break

    final_profit = profit / 50
    if minimize:
        final_profit *= -1
    return final_profit


if __name__ == "__main__":
    # Example action_vector, adjust as needed
    action_vector = [2, 2, 2, 2, 2, 2, 2, 2]
    obj = 0  # or 1 for minimization
    result = function_value(action_vector, obj)
    print("Result:", result)
