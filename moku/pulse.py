import numpy as np
import math 
import time
import sys
import yaml
from moku.instruments import WaveformGenerator
from datetime import datetime


#Ask user to input IP address of Moku device
ip_flag = True
while ip_flag:
    ip = '[' + input('Please enter the IP address of the moku device you would like to connect to: ') + ']'
    #ip = '[fe80::cb04:35db:d5c2:f05d]'
    try: 
        # Connect to your Moku by its ip address ip
        # force_connect will overtake an existing connection
        i = WaveformGenerator(ip, force_connect=True)
        ip_flag = False
    except Exception as e:
        print("An error while trying to connect to the IP address you provided (",ip,f"): {e}")

#Channel number set to 4 for MokuPro. Add functionality to automatically set channel number. 
channel_no = 4

#Set default parameters
amp = 1 #Starting amplitude 
freq = 1 #Repetition rate of pulse
pulse_width = 0.1 #Pulse width
edge_width = 1e-8 #Pulse edge with, I don't think this can be set to 0
no_pulses = 10 #Number of pulses (negative for infinite sequence)


#Read conf_pulse.yaml file for parameters, output signal duration and number of pulses,
# and ask use for confirmation before producing signal
check_sig = True
while check_sig:
    try:

        #Read configuration from yaml file
        try: 
            with open('conf_pulse.yaml', 'r') as file:
                loaded_conf = yaml.safe_load(file)

            print("\nData read from 'conf_pulse.yaml':")
            [print(f"{key}: {value}") for key, value in loaded_conf.items()]
            amp = loaded_conf['amplitude']
            freq = loaded_conf['repetition rate']
            pulse_width = loaded_conf['pulse width']
            edge_width = loaded_conf['edge width']
            no_pulses = loaded_conf['no_pulses']

            amp = float(amp)
            freq = float(freq)
            pulse_width = float(pulse_width)
            edge_width = float(edge_width)
            no_pulses = int(no_pulses)

        except Exception as e:
            print(f"An error occurred in trying to load conf_pulse.yaml: {e}")
            print('Using default parameters.')
        
        max_time = no_pulses*(1/freq) #Calculate max_time

        #Output signal duration and number of pulses and ask for user confirmation 
        if no_pulses >= 0:
            foo = input('\nTotal signal duration is ' + str(max_time) + 's, with ' + str(no_pulses) + 
                    ' pulses. Enter y to proceed or upload new conf_pulse.yaml file\n')
        elif no_pulses < 0: 
            foo = input('\nContinuous string of pulses requested. Enter y to proceed or upload new conf_pulse.yaml file\n')
        else:
            foo = input('\nNo pulse requested. Enter y to proceed or upload new conf_pulse.yaml file\n')
        
        if foo == 'y':
            check_sig = False
    
    except Exception as e:
            print(f"An error occurred: {e}")
        

#Store configuration to conf dict
conf = {
    'amplitude': amp,
    'repetition rate': freq,
    'pulse width': pulse_width,
    'edge width': edge_width,
    'no_pulses': no_pulses,
}



#Put date, time etc. in conf file name
file_name = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss") + "_conf_pulse.yaml"

#Save conf as yaml file
with open(file_name,'w') as file:
    yaml.dump(conf, file)
print('Configuration has been saved to ' + file_name + '\n')



try:
    
    if no_pulses == 0:
        print('no_pulses set to 0. No pulse produced. Please input a negative interger for a continuous pulse sequence.')

    elif no_pulses != 0:
        #Offset set to amp/2 so the pulse is from 0V 

        i.set_output_termination(channel_no, 'HiZ')
        i.generate_waveform(channel=channel_no, type='Pulse', amplitude=amp, 
                            pulse_width = pulse_width, edge_time = edge_width, frequency=freq, offset=amp/2)
    
        print('Printing summary of initial state: \n')
        print(i.summary())

        pulse_index = np.sign(no_pulses)
        while pulse_index <= abs(no_pulses):
        
            if no_pulses >= 0:
                print('Running... Press Ctrl + C to stop. Pulse ',pulse_index,' of ', no_pulses)
            
            else:
                print('Pulse no.',abs(pulse_index),'. Running continuously until interrupted... Press Ctrl + C to stop.')

            time.sleep((1/freq)*0.5)
            
            if pulse_index == abs(no_pulses):
                #print('Running... Press Ctrl + C to stop. Pulse ',pulse_index,' of ', no_pulses)
                i.generate_waveform(channel=channel_no, type='Off')
                
            time.sleep((1/freq)*0.5) #The timing may eventually get thrown off for very
                                        #long pulse signals 
        
            pulse_index += np.sign(no_pulses)

        print(pulse_index-1, ' of ', no_pulses,' pulses produced. ', max_time,' s elasped. Terminating program.')

except KeyboardInterrupt:
    print("Program to be terminated at user's request.\n")

finally:
    #A little laggy here, pulse continues for a little longer than desired before shutting off. 
    i.generate_waveform(channel=channel_no, type='Off')
    #i.generate_waveform(channel=1, type='Off')

    print('Printing endstate summary: \n')
    print(i.summary())

    i.relinquish_ownership()
    sys.exit("Program terminated.")


