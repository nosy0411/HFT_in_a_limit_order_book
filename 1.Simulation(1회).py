# 마켓메이킹의 재고관리 전략 Simulation - 1회 simulation
#
# Marco Avellaneda and Shasha Stoikov, 2006, High-frequency trading in a limit order book
# 원 코드 출처 : https://github.com/ragoragino/avellaneda-stoikov
#
# ----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Parameters of the simulation
s0 = 100        # 초기주가
T = 1           # expiration (t = 0 ~ T). T = 1 (day)
sigma = 2       # mid-price 변화율
dt = T / 200    # time 구간 (0, dt, dt+dt, ... , T) 0 ~ 1 사이를 200 등분한다. dt = 0.005
q0 = 0          # 초기 재고
gamma = 0.1     # 마켓메이커의 위험회피 정도
ak = 1.5        # alpha * 유동성 척도
alpha = 1.4     # market order 분포 parameter
lambdaL = 196   # market order frequency a day (T) : dividing the total volume traded 
                # over a day by the average size of market orders on that day
A = lambdaL / alpha

# s1 = 재고관리 전략 (asymmetric)
inventory_s1 = q0
pnl_s1 = 0

# s2 = 그냥 대칭전략 (symmetric)
inventory_s2 = q0
pnl_s2 = 0

# Variables holding the price properties that will be used in the price plot
price_a = [0] * (int(T / dt) + 1)
price_b = [0] * (int(T / dt) + 1)
midprice = [0] * (int(T / dt) + 1)

# Optimal spread. 대칭 전략도 이 스프레드들 사용한다. optimal_spread = 1.29
optimal_spread = 2 * np.log(1 + gamma / ak) / gamma

# dt 동안 지정가 주문 체결 횟수 (intensity). 이 식을 내 지정가 주문이 dt 동안 체결될 확률로 사용할 수 있다.
prob = A * np.exp(- ak * optimal_spread / 2) * dt

# Simulation
profitTraj_s1 = []
invTraj_s1 = []
profitTraj_s2 = []
invTraj_s2 = []

# 주가 (mid-price) 시뮬레이션. 랜덤워크.
# The mid-price is updated by a random increment +-sigma * sqrt(dt).
white_noise = sigma * np.sqrt(dt) * np.random.choice([1, -1], int(T / dt))
price_process = s0 + np.cumsum(white_noise)
price_process = np.insert(price_process, 0, s0)
plt.plot(price_process)
plt.show()

for step, s in enumerate(price_process):
    # 유보가격 (reservation_price)을 계산하고, 유보가격을 기준으로 스프레드를 계산한다. r에 대해 대칭.
    reservation_price = s - inventory_s1 * gamma * sigma**2 * (T - step * dt)
    spread = np.log(1 + gamma / ak) / gamma
    
    # 시장 가격 (s)을 기준으로 ask, bid 스프레드를 계산한다. inventory_s1에 따라 비대칭이 된다.
    if reservation_price >= s:
        # ex : s = 100, r = 110, spread = 20 --> ask_spread = 30, bid_spread = 10
        # ask_price = s + ask_sparead = 130, bid_price = 90
        ask_spread = spread + (reservation_price - s)
        bid_spread = spread - (reservation_price - s)
    else:
        # ex : s = 100, r = 90, spread = 20 --> ask_spread = 10, bid_spread = 30
        # ask_price = s + ask_sparead = 110, bid_price = 70
        ask_spread = spread - (s - reservation_price)
        bid_spread = spread + (s - reservation_price)

    # 주문을 처리한다
    if ask_spread > 0 and bid_spread > 0:
        # 지정가 주문 처리
        ask_prob = A * np.exp(- ak * ask_spread) * dt
        bid_prob = A * np.exp(- ak * bid_spread) * dt
        ask_prob = max(0, min(ask_prob, 1))
        bid_prob = max(0, min(bid_prob, 1))
        ask_action_s1 = np.random.choice([1, 0], p=[ask_prob, 1 - ask_prob])
        bid_action_s1 = np.random.choice([1, 0], p=[bid_prob, 1 - bid_prob])

        inventory_s1 -= ask_action_s1                # ask 측 주문이 체결되었음 (매도 재고 증가)
        pnl_s1 += ask_action_s1 * (s + ask_spread)   # profit 계산
        inventory_s1 += bid_action_s1                # bid 측 주문이 체결되었음 (매수 재고 증가)
        pnl_s1 -= bid_action_s1 * (s - bid_spread)   # profit 계산
    elif ask_spread < 0 and bid_spread > 0:
        # Ask 측은 시장가 주문 처리, Bid 측은 지정가 주문 처리
        bid_prob = A * np.exp(- ak * bid_spread) * dt
        bid_prob = max(0, min(bid_prob, 1))
        bid_action_s1 = np.random.choice([1, 0], p=[bid_prob, 1 - bid_prob])

        inventory_s1 -= 1                            # 시장가 매도 주문이 체결되었음 (매도 재고 증가)
        pnl_s1 += 1 * (s + ask_spread)               # profit 계산
        
        inventory_s1 += bid_action_s1                # bid 측 주문이 체결되었음 (매수 재고 증가)
        pnl_s1 -= bid_action_s1 * (s - bid_spread)   # profit 계산
    elif ask_spread > 0 and bid_spread < 0:
        # Ask 측은 지정가 주문 처리, Bid 측은 시장가 주문 처리
        ask_prob = A * np.exp(- ak * ask_spread) * dt
        ask_prob = max(0, min(ask_prob, 1))
        ask_action_s1 = np.random.choice([1, 0], p=[ask_prob, 1 - ask_prob])

        inventory_s1 -= ask_action_s1                # ask 측 주문이 체결되었음 (매도 재고 증가)
        pnl_s1 += ask_action_s1 * (s + ask_spread)   # profit 계산
        
        inventory_s1 += 1                            # bid 측 주문이 체결되었음 (매수 재고 증가)
        pnl_s1 -= 1 * (s - bid_spread)               # profit 계산
    else:
        # 모두 시장가 주문 처리
        inventory_s1 -= 1                            # ask 측 주문이 체결되었음 (매도 재고 증가)
        pnl_s1 += 1 * (s + ask_spread)               # profit 계산
        
        inventory_s1 += 1                            # bid 측 주문이 체결되었음 (매수 재고 증가)
        pnl_s1 -= 1 * (s - bid_spread)               # profit 계산
        
    # Symmetric strategy
    ask_action_s2 = np.random.choice([1, 0], p=[prob, 1 - prob])
    bid_action_s2 = np.random.choice([1, 0], p=[prob, 1 - prob])
    inventory_s2 -= ask_action_s2
    pnl_s2 += ask_action_s2 * (s + optimal_spread / 2)
    inventory_s2 += bid_action_s2
    pnl_s2 -= bid_action_s2 * (s - optimal_spread / 2)
    
    price_a[step] = s + ask_spread
    price_b[step] = s - bid_spread
    midprice[step] = s
    profitTraj_s1.append(pnl_s1)
    invTraj_s1.append(inventory_s1)
    profitTraj_s2.append(pnl_s2)
    invTraj_s2.append(inventory_s2)

# 남은 재고는 mid-price에 청산한다.
pnl_s1 += inventory_s1 * s
pnl_s2 += inventory_s2 * s

print("check1")
x = np.arange(0, T + dt, dt)
plt.figure(figsize=(12, 6))
plt.plot(x, price_a, linewidth=1.0, linestyle="-", label="ASK")
plt.plot(x, price_b, linewidth=1.0, linestyle="-", label="BID")
plt.plot(x, midprice, linewidth=3, linestyle="-", label="MID-PRICE")
plt.legend()
plt.title("The mid-price and the optimal bid and ask quotes")
plt.show()
print("check2")
plt.figure(figsize=(12, 6))
plt.plot(x, profitTraj_s1, linewidth=1.0, linestyle="-")
plt.title("Inventory Strategy : Cumulative Profit")
plt.show()
print("check3")
plt.figure(figsize=(12, 6))
plt.plot(x, invTraj_s1, linewidth=1.0, linestyle="-")
plt.title("Inventory Strategy : Cumulative Inventory")
plt.show()
print("check4")
plt.figure(figsize=(12, 6))
plt.plot(x, profitTraj_s2, linewidth=1.0, linestyle="-")
plt.title("Symmetric Strategy : Cumulative Profit")
plt.show()
print("check5")
plt.figure(figsize=(12, 6))
plt.plot(x, invTraj_s2, linewidth=1.0, linestyle="-")
plt.title("Symmetric Strategy : Cumulative Inventory")
plt.show()
print("check6")
    