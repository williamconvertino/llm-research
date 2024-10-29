import time

def calculate_time_remaining(start_time, i, n):
  elapsed_time = time.time() - start_time
  time_remaining = elapsed_time * (n - i) / (i + 1)
  time_remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
  return time_remaining_formatted