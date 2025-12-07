def computeFee(basePrice, n_passengers, n_over18, n_under15):
    # Check if the total number of passengers is more than 5
    if n_passengers > 5:
        raise ValueError("Groups cannot have more than 5 passengers.")
    
    # Check if the group meets the criteria for the free ride offer
    if n_over18 >= 1 and n_under15 >= 2:
        return 0.0
    
    # Calculate the fee for the remaining passengers who do not qualify for the free ride
    remaining_passengers = n_passengers - n_over18 - n_under15
    fee = remaining_passengers * basePrice
    
    return fee