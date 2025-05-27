
```julia
# Grazing Probability Calculator
# Shows probability of grazing behavior at different temperature and time combinations

# Parameters from model
temp_mean = 2.04
temp_std = 6.21
time_mean = 768.29
time_std = 237.66

# Temperatures and times to test
test_temps = [-10, -5, 0, 5, 10, 15]
test_hours = [6, 8, 10, 12, 14, 16, 18, 20, 22]

# Calculate probability function
function calc_grazing_prob(temp, hour)
    temp_z = (temp - temp_mean) / temp_std
    time_z = (hour * 60 - time_mean) / time_std
    
    logit = -0.284 - 0.239*temp_z - 0.065*time_z + 2.467*time_z^2 + 
            1.494*time_z^3 - 1.238*time_z^4 - 0.26*temp_z*time_z + 
            0.159*temp_z*time_z^4
    
    return 1 / (1 + exp(-logit))
end

# # Probability of Grazing Behavior (%)
# | Temp (°C) |  6:00 |  8:00 | 10:00 | 12:00 | 14:00 | 16:00 | 18:00 | 20:00 | 22:00 |
# |-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
# | -10°C     |  <1 % |  6 %  |  55 % |  54 % |  64 % |  91 % |  98 % |  77 % |  <1 % |
# | -5°C      |  <1 % |  8 %  |  54 % |  51 % |  58 % |  88 % |  98 % |  89 % |  <1 % |
# |  0°C      |  <1 % |  11 % |  54 % |  47 % |  51 % |  84 % |  97 % |  95 % |  <1 % |
# |  5°C      |  <1 % |  15 % |  54 % |  43 % |  45 % |  79 % |  97 % |  98 % |   7 % |
# |  10°C     |  <1 % |  20 % |  53 % |  40 % |  39 % |  74 % |  97 % |  99 % |  61 % |
# |  15°C     |  <1 % |  26 % |  53 % |  36 % |  33 % |  68 % |  97 % |  100% |  97 % |


# Print table header
println("# Probability of Grazing Behavior (%)")
print("| Temp (°C) |")
for hour in test_hours
    print(" $(hour):00 |")
end
println()

# Print table separator
print("|-----------|")
for _ in test_hours
    print("-------|")
end
println()

# Print table rows
for temp in test_temps
    print("""| $(temp < 0 ? "" : " ")$(rpad(string(temp,"°C"),4))     |""")
    for hour in test_hours
        prob = calc_grazing_prob(temp, hour) * 100
        prob_text = prob < 1 ? "<1" : round(Int, prob)
        print(" $(prob_text)$(prob_text == 100 ? "" : " ")% |")
    end
    println()
end

# Probability of grazing (temp = 7.0, hour = 12.8): 38.3%
# Probability of grazing (temp = -10.0, hour = 12.8): 54.5%
# Probability of grazing (temp = -10.0, hour = 20): 100.0%
```