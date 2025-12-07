def computeMaxTime(winner_time, avg_speed, track_type):
    # Check for valid inputs
    if not isinstance(winner_time, (int, float)) or winner_time < 0:
        return 0
    if not isinstance(avg_speed, (int, float)) or avg_speed < 0:
        return 0
    if track_type not in ['A', 'B', 'C']:
        return 0
    
    # Compute the maximum time based on the track type and average speed
    if track_type == 'A':
        if avg_speed <= 30:
            max_time = winner_time * 1.05
        elif avg_speed <= 35:
            max_time = winner_time * 1.10
        else:
            max_time = winner_time * 1.15
    elif track_type == 'B':
        if avg_speed <= 30:
            max_time = winner_time * 1.20
        elif avg_speed <= 35:
            max_time = winner_time * 1.25
        else:
            max_time = winner_time * 1.30
    else:  # track_type == 'C'
        max_time = winner_time * 1.50
    
    return round(max_time, 2)