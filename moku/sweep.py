import sys
import yaml
from moku.instruments import WaveformGenerator

import sweep_core
import sweep_limits

# The conf parsing, ramp planning and drive loop live in sweep_core.py so the
# particle loader (dropper_server.py) can reuse them; this file is the
# interactive CLI and behaves as it always has.

format_ip = sweep_core.format_ip

#Load IP address from config file, or ask user to input it
ip = None
try:
    with open('conf_sweep.yaml', 'r') as file:
        conf_ip = yaml.safe_load(file)
    if 'ip_address' in conf_ip and conf_ip['ip_address']:
        ip = format_ip(conf_ip['ip_address'])
        print(f"Loaded IP address from conf_sweep.yaml: {conf_ip['ip_address']}")
except Exception as e:
    print(f"Could not read IP from conf_sweep.yaml: {e}")

ip_flag = True
while ip_flag:
    if ip is None:
        ip = format_ip(input('Please enter the IP address of the moku device you would like to connect to: '))
    try:
        # Connect to your Moku by its ip address ip
        # force_connect will overtake an existing connection
        i = WaveformGenerator(ip, force_connect=True)
        ip_flag = False
    except Exception as e:
        print("An error while trying to connect to the IP address you provided (",ip,f"): {e}")
        ip = None


#Read conf_sweep.yaml file for parameters, output signal duration and number of pulses,
# and ask use for confirmation before producing signal
check_sig = True
while check_sig:
    try:

        #Read configuration from yaml file
        try:
            loaded_conf = sweep_core.load_conf('conf_sweep.yaml')
            print("\nData read from 'conf_sweep.yaml':")
            [print(f"{key}: {value}") for key, value in loaded_conf.items()]
        except Exception as e:
            print(f"An error occurred in trying to load conf_sweep.yaml: {e}")
            print('Using default parameters.')
            loaded_conf = {}

        #Calculate time of signal and no_pulses (or amp_incr) -- see sweep_core.plan_ramp
        plan = sweep_core.plan_ramp(loaded_conf)
        if loaded_conf.get('amplitude increment', 1.01) not in (None, 'None'):
            print('Calculated number of pulses is:', plan['no_pulses'])
        else:
            print('Calculated multiplicative amplitude increment is:',
                  round(plan['amplitude increment'], 3))

        #Check the whole planned ramp against the Moku:Go +-5 V output range before
        #any output. A DC offset that fits on pulse 1 can exceed the rail hundreds
        #of pulses later, unattended, so warn here rather than discovering it mid-run.
        #Warning only: proceeding deliberately is still allowed.
        headroom = plan['headroom']
        print('\nDC offset: {:+g} V. Peak output excursion over the ramp: {:.3f} V '
              '(limit +-{:g} V).'.format(plan['dc offset'], headroom['peak_v'],
                                         sweep_limits.PEAK_LIMIT_V))
        for msg in headroom['messages']:
            print('WARNING: ' + msg)

        #Output signal duration and number of pulses and ask for user confirmation
        foo = input('\nTotal signal duration is ' + str(plan['max_time']) + 's, with ' +
                    str(plan['no_pulses']) +
                    ' pulses. Enter y to proceed or upload new conf_sweep.yaml file\n')
        if foo == 'y':
            check_sig = False

    except Exception as e:
            print(f"An error occurred: {e}")

plan['ip_address'] = ip.strip('[]')

#Put date, time etc. in conf file name and save to Dropbox data directory
file_path = sweep_core.save_conf_copy(plan)
print('Configuration has been saved to ' + file_path + '\n')


def on_pulse(k, c_amp, time_elapsed):
    if time_elapsed == 0:
        print('Printing summary of initial state: \n')
        print(i.summary())
    T = plan['sweep duration']
    pulse_index = int(round((plan['no_pulses']*(time_elapsed + T))/plan['max_time']))
    print('Running... Press Ctrl + C to stop. Pulse amplitude:', round(c_amp,3),
          ' V. DC offset:', round(plan['dc offset'],3), ' V. Pulse ',pulse_index,' of ',
          plan['no_pulses'])


try:
    sweep_core.run_sweep(i, plan, on_pulse=on_pulse)
    print('\nSignal generation complete. Terminating program.')

except Exception as e:
    print(f"An unexpected error occurred: {e}")
except KeyboardInterrupt:
    print("Program to be terminated at user's request.\n")

finally:
    #set_defaults() alone is laggy (the pulse continues for a while), so switch
    #the waveform Off first, then restore defaults.
    try:
        sweep_core.output_off(i, plan['channel'])
    except Exception as e:
        print(f"Could not switch output off directly: {e}")
    i.set_defaults()

    print('Printing endstate summary: \n')
    print(i.summary())

    i.relinquish_ownership()
    sys.exit("Program terminated.")
