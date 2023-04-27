# The rain values prior to 2011 appear to be null when they should be 0

f = open('dirty_input.csv', 'r')
line_count = 0
bad_lines = 0
zero_rain_lines = 0
with open('clean_input.csv', 'w') as output:
    for line in f:
        line_count += 1
        tokens = line.strip().split(",")
        
        found_bad = False
        
        reconstructed_line = ''
        for i in range(0, len(tokens)):
            if i > 0:
                reconstructed_line = reconstructed_line + ','
        
            if(i == len(tokens)-1):
                if 'null' in tokens[i]:
                    reconstructed_line = reconstructed_line + '0.00'
                    zero_rain_lines += 1
                else:
                    reconstructed_line = reconstructed_line + tokens[i]
                    
                if tokens[i] == '0' or tokens[i] == '0.00':
                    zero_rain_lines += 1
            else:
                if 'null' in tokens[i]:
                    found_bad = True
                    bad_lines += 1
                    break;
                reconstructed_line = reconstructed_line + tokens[i]
        
        if not found_bad:
            output.write(reconstructed_line+"\n")

f.close()
print("Total Lines: " + str(line_count))
print("Bad lines: " + str(bad_lines))
print("Good lines: " + str(line_count - bad_lines))
zero_rain_pct = round((zero_rain_lines / line_count) * 100, 2)
print("Lines with zero rain: " + str(zero_rain_lines) + ", " + str(zero_rain_pct)+"%")