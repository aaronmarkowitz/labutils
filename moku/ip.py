#test comment 
import subprocess
import yaml as yaml 

#Run mokucli list in command line and store output
output = subprocess.run("mokucli list", capture_output=True, text=True)

#Split output into words and remove header and formatting
elements = output.stdout.split()
device_elements = elements[7:]

#Prepare dictionary 
device_ip_dict = {}

#Add names, hardware ids, and ip addresses to dict in order provided by mokucli list
for k in range(len(device_elements)):
    if k%6 == 0:
        index = int(k/6 + 1)

        ip = device_elements[k+4]

        #Remove portion after and including % symbol at the end of IP address
        pos = ip.find('%')
        if pos != -1:
            ip = ip[:pos]

        device_ip_dict[str(index)] = [device_elements[k], device_elements[k+2], ip]

#Proceed only if dictionary is not empty i.e. at least one device is found
if bool(device_ip_dict):

    #Output Moku devices and IP addresses in an ordered list
    print('Moku devices and IP addresses are as follows:\n')
    [print(f"{key}. {value[0]}: {value[2]} (Device Type: {value[1]})") for key, value in device_ip_dict.items()]

    #Ask user to select a device and store the corresponding IP address
    check = True
    if len(device_ip_dict.keys()) == 1:
        check = False
        print('Only one device found.\n')
        x = 1

    while check: 
        x = input('Please type in the number of the moku device you would like to connect to: ')
        try: 
            if int(x) <= len(device_elements)/5 and int(x) > 0:
                check = False

            else:
                print('Invalid number, please pick an index from the list of devices. \n')

        except Exception as e:
            print(f"An error occurred: {e}")

    
    #Store configuration to conf dict
    moku_inf = {
        'name': device_ip_dict[str(x)][0],
        'device type': device_ip_dict[str(x)][1],
        'IP address': device_ip_dict[str(x)][2],
    }

    print('Storing the following device details:\n')

    [print(f"\t{key}: {value}\n") for key, value in moku_inf.items()]


    #Save conf as yaml file
    with open('moku_device_info.yaml','w') as file:
        yaml.dump(moku_inf, file)
    print('Configuration has been saved to moku_device_info.yaml\n')


#Otherwise state that no devices have been found
else:
    print('No devices found.')




