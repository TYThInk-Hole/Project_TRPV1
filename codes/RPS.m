using Plots
using DelimitedFiles
# Define the file path
file_path = "/Volumes/yoonJL/RPS/data/200/5/inter/30/dth/inter_th.txt"    

# Open the file for reading
file = open(file_path, "r")

# Keep track of how much data has been read
last_position = position(file)

# Real-time monitoring loop
# while true
    # Seek to the last read position
    seek(file, last_position)
    
    # Read new data from the file
    new_data = readdlm(file_path, ',')
    
#plot
plot!(new_data[1,:],new_data[6,:])

    # If new data exists, process it
    if !isempty(new_data)
        println("New data detected:")
        println(new_data)
    end
    
    sleep(1)  # Adjust the interval as needed
# end

# Close the file (if needed, though in a long-running script this might be omitted)
close(file)