import numpy as np
import math 
import time
import sys
import yaml
from moku.instruments import WaveformGenerator
from datetime import datetime


#Load IP address from config file, or ask user to input it
ip = None
try:
    with open('conf_sweep.yaml', 'r') as file:
        conf_ip = yaml.safe_load(file)
    if 'ip_address' in conf_ip and conf_ip['ip_address']:
        ip = '[' + conf_ip['ip_address'] + ']'
        print(f"Loaded IP address from conf_sweep.yaml: {conf_ip['ip_address']}")
except Exception as e:
    print(f"Could not read IP from conf_sweep.yaml: {e}")

ip_flag = True
while ip_flag:
    if ip is None:
        ip = '[' + input('Please enter the IP address of the moku device you would like to connect to: ') + ']'
    try:
        # Connect to your Moku by its ip address ip
        # force_connect will overtake an existing connection
        i = WaveformGenerator(ip, force_connect=True)
        ip_flag = False
    except Exception as e:
        print("An error while trying to connect to the IP address you provided (",ip,f"): {e}")
        ip = None



#Set default parameters
channel_no = 1 #Default channel number
amp = 1 #Starting amplitude of waveform
end_amp = 2 #Ending amplitude of waveform 
start_freq = 1 #Starting frequency
stop_freq = 100 #Ending frequency
T = 3 #duration of sweep
amp_incr = 1.01 #Multiplicative amplitude increment
no_pulses = 10 #Number of pulses (note: redundant as amp_incr and the other variables determine 
                #the number of pulses)



#Read conf_sweep.yaml file for parameters, output signal duration and number of pulses,
# and ask use for confirmation before producing signal
check_sig = True
while check_sig:
    try:

        #Read configuration from yaml file
        try: 
            with open('conf_sweep.yaml', 'r') as file:
                loaded_conf = yaml.safe_load(file)

            print("\nData read from 'conf_sweep.yaml':")
            [print(f"{key}: {value}") for key, value in loaded_conf.items()]
            amp = loaded_conf['amp']
            end_amp = loaded_conf['end amp']
            start_freq = float(loaded_conf['base frequency'])
            stop_freq = float(loaded_conf['stop frequency'])
            T = loaded_conf['sweep duration']
            amp_incr = loaded_conf['amplitude increment']
            if amp_incr == 'None':
                amp_incr = None
            no_pulses = loaded_conf['no_pulses']
            channel_no = int(loaded_conf.get('channel', channel_no))

        except Exception as e:
            print(f"An error occurred in trying to load conf_sweep.yaml: {e}")
            print('Using default parameters.')
        
        #Calculate time of signal and no_pulses if amp_incr is given
        if amp_incr != None:
            max_pulses = np.log(end_amp/amp)/np.log(amp_incr)
            no_pulses = int(math.ceil(max_pulses)) + 1
            print('Calculated number of pulses is:', no_pulses)
            max_time = T*no_pulses

        #If amp_incr is not given, calculate time of signal and amp_incr using no_pulses
        else:
            max_time = no_pulses*T
            amp_incr = (end_amp/amp)**(1/(no_pulses-1))
            print('Calculated multiplicative amplitude increment is:', round(amp_incr,3))

        #Output signal duration and number of pulses and ask for user confirmation 
        foo = input('\nTotal signal duration is ' + str(max_time) + 's, with ' + str(no_pulses) + 
                    ' pulses. Enter y to proceed or upload new conf_sweep.yaml file\n')
        if foo == 'y':
            check_sig = False
    
    except Exception as e:
            print(f"An error occurred: {e}")
        

#Store configuration to conf dict
conf = {
    'amp': amp,
    'channel': channel_no,
    'end amp': end_amp,
    'ip_address': ip.strip('[]'),
    'base frequency': start_freq,
    'stop frequency': stop_freq,
    'sweep duration' : T,
    'amplitude increment': amp_incr,
    'no_pulses': no_pulses,
}


#Put date, time etc. in conf file name
file_name = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss") + "_conf_sweep.yaml"

#Save conf as yaml file
with open(file_name,'w') as file:
    yaml.dump(conf, file)
print('Configuration has been saved to ' + file_name + '\n')



try:
    # Load and configure the waveform.
    #i.generate_waveform(channel=1, type='Sine', amplitude=amp, frequency=start_freq)

    #Don't think you can do modulation and sweep mode at the same time, so I don't think you can do amplitude modulation 
    #i.set_modulation(channel=1, type='Amplitude', source='Internal', depth=10, frequency=10)

    #i.set_sweep_mode(channel=1, source='Internal', stop_frequency=stop_freq, sweep_time=T, trigger_level=0)

    time_elapsed = 0
    c_amp = amp
    
    # Enable output channel 1 is not required, it is enabled by the generate_waveform call.
    #i.enable_outputs(channel=1, enable=True)
    
    while time_elapsed <= max_time:

        i.generate_waveform(channel=channel_no, type='Sine', amplitude=c_amp, frequency=start_freq)
        i.set_sweep_mode(channel=channel_no, source='Internal', stop_frequency=stop_freq, sweep_time=T, trigger_level=0)

        if time_elapsed == 0:
            print('Printing summary of initial state: \n')
            print(i.summary())

        pulse_index = int(round((no_pulses*(time_elapsed + T))/max_time))
        print('Running... Press Ctrl + C to stop. Pulse amplitude:', round(c_amp,3), ' V. Pulse ',pulse_index,' of ', no_pulses)
        time.sleep(T)
        time_elapsed += T
        c_amp = c_amp*amp_incr
    
    print('\nSignal generation complete. Terminating program.')
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")
except KeyboardInterrupt:
    print("Program to be terminated at user's request.\n")

finally:
    #A little laggy here, pulse continues for a little longer than desired before shutting off. 
    i.set_defaults()

    print('Printing endstate summary: \n')
    print(i.summary())

    i.relinquish_ownership()
    sys.exit("Program terminated.")

