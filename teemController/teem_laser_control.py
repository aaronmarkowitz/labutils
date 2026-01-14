#!/home/controls/.conda/envs/controls/bin/python3
"""
Teem Laser Control Script
Author: Claude Sonnet 4.5
Date: 2026-01-13

User-friendly script to control the Teem Photonics laser via EPICS IOC channels.

Usage:
  teem_laser_control.py on <duration>    # Turn on for N seconds
  teem_laser_control.py on -1            # Continuous (until Ctrl+C)
  teem_laser_control.py off              # Turn off immediately
  teem_laser_control.py status           # Show laser status
"""

import argparse
import time
import signal
import sys

try:
    from epics import caget, caput
except ImportError:
    print("ERROR: pyepics not installed. Install with: pip install pyepics")
    sys.exit(1)

PREFIX = "Y1:AUX-"


class LaserControl:
    """Laser control with deadman switch"""

    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C or kill - ensures laser turns off"""
        print("\n\nShutdown signal received - turning off laser...")
        self.running = False

    def wait_for_connection(self, timeout=5.0):
        """Wait for EPICS IOC to be available"""
        print("Checking IOC connection...")
        start = time.time()

        while time.time() - start < timeout:
            try:
                result = caget(PREFIX + 'UVD_LASER_STATE', timeout=1.0)
                if result is not None:
                    print("IOC connected")
                    return True
            except:
                pass
            time.sleep(0.5)

        print("ERROR: Cannot connect to IOC")
        print(f"Make sure auxioc service is running: systemctl status auxioc")
        return False

    def check_service_running(self):
        """Check if teem-laser service is running"""
        uptime = caget(PREFIX + 'UVD_UPTIME', timeout=1.0)

        if uptime is None or uptime == 0:
            print("WARNING: teem-laser service may not be running")
            print("Start with: sudo systemctl start teem-laser")
            print("Check status: systemctl status teem-laser")
            return False
        return True

    def get_status(self) -> dict:
        """Get current laser status"""
        state_map = {0: 'OFF', 1: 'STARTING', 2: 'ON', 3: 'STOPPING', 4: 'ERROR'}

        status = {
            'state': state_map.get(caget(PREFIX + 'UVD_LASER_STATE'), 'UNKNOWN'),
            'emitting': bool(caget(PREFIX + 'UVD_EMITTING')),
            'ready': bool(caget(PREFIX + 'UVD_READY')),
            'temp_ok': bool(caget(PREFIX + 'UVD_TEMP_OK')),
            'diode_temp': caget(PREFIX + 'UVD_DIODE_TEMP'),
            'crystal_temp': caget(PREFIX + 'UVD_CRYSTAL_TEMP'),
            'heatsink_temp': caget(PREFIX + 'UVD_HEATSINK_TEMP'),
            'emission_hours': caget(PREFIX + 'UVD_EMISSION_HOURS'),
            'emission_minutes': caget(PREFIX + 'UVD_EMISSION_MINUTES'),
            'heartbeat_timeout': caget(PREFIX + 'UVD_HEARTBEAT_TIMEOUT'),
            'last_error': caget(PREFIX + 'UVD_LAST_ERROR'),
        }

        return status

    def print_status(self):
        """Print formatted status"""
        status = self.get_status()

        print("\n" + "="*50)
        print("TEEM PHOTONICS LASER STATUS")
        print("="*50)
        print(f"State:           {status['state']}")
        print(f"Emitting:        {'YES' if status['emitting'] else 'NO'}")
        print(f"Ready:           {'YES' if status['ready'] else 'NO'}")
        print(f"Temp OK:         {'YES' if status['temp_ok'] else 'NO'}")
        print(f"\nTemperatures:")
        print(f"  Diode:         {status['diode_temp']:.2f} °C")
        print(f"  Crystal:       {status['crystal_temp']:.2f} °C")
        print(f"  Heatsink:      {status['heatsink_temp']:.0f} °C")
        print(f"\nRuntime:")
        print(f"  Emission:      {status['emission_hours']}h {status['emission_minutes']}m")
        print(f"\nSafety:")
        print(f"  Heartbeat:     {status['heartbeat_timeout']:.1f}s timeout")

        if status['last_error']:
            print(f"\nLast Error:      {status['last_error']}")

        print("="*50 + "\n")

    def turn_on(self, duration: float):
        """
        Turn on laser and keep alive with heartbeat

        Args:
            duration: Duration in seconds (-1 for continuous)
        """
        if not self.wait_for_connection():
            return False

        if not self.check_service_running():
            return False

        # Check current state
        current_state = caget(PREFIX + 'UVD_LASER_STATE')
        if current_state == 2:  # Already ON
            print("WARNING: Laser is already ON")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                return False

        print("\n" + "="*50)
        print("STARTING LASER")
        print("="*50)

        # Get heartbeat settings
        timeout = caget(PREFIX + 'UVD_HEARTBEAT_TIMEOUT')
        heartbeat_interval = timeout / 4.0  # Send at 4x the timeout rate

        print(f"Heartbeat timeout:  {timeout:.1f}s")
        print(f"Heartbeat interval: {heartbeat_interval:.2f}s")

        if duration > 0:
            print(f"Duration:           {duration:.1f}s")
        else:
            print("Duration:           Continuous (Ctrl+C to stop)")

        print("\nSending start command...")

        # Send start command and initial heartbeat
        caput(PREFIX + 'UVD_LASER_ON', 1, wait=True)
        caput(PREFIX + 'UVD_TURN_OFF', 0, wait=False)  # Initial heartbeat

        # Wait for ready state (up to 10s)
        print("Waiting for laser to be ready...")
        for i in range(100):
            if not self.running:
                print("Interrupted during startup")
                self.turn_off()
                return False

            # Send heartbeat every iteration to prevent deadman timeout
            caput(PREFIX + 'UVD_TURN_OFF', 0, wait=False)

            state = caget(PREFIX + 'UVD_LASER_STATE')
            ready = caget(PREFIX + 'UVD_READY')

            if state == 4:  # ERROR
                print("\nERROR: Laser entered error state")
                last_error = caget(PREFIX + 'UVD_LAST_ERROR')
                if last_error:
                    print(f"Error: {last_error}")
                self.print_status()
                return False

            if state == 2 and ready:  # ON and READY
                break

            time.sleep(0.1)
        else:
            print("\nERROR: Laser not ready after 10s")
            self.print_status()
            self.turn_off()
            return False

        emitting = caget(PREFIX + 'UVD_EMITTING')
        if not emitting:
            print("\nWARNING: Laser ready but not emitting")
            print("Check error registers and manual front panel")

        print("\n" + "="*50)
        print("LASER ON - Maintaining heartbeat")
        print("="*50)
        print("Press Ctrl+C to stop\n")

        # Heartbeat loop
        start_time = time.time()
        last_status_time = start_time

        while self.running:
            # Send heartbeat
            caput(PREFIX + 'UVD_TURN_OFF', 0, wait=False)

            # Check for errors
            state = caget(PREFIX + 'UVD_LASER_STATE')
            if state == 4:  # ERROR
                print("\nERROR: Laser entered error state")
                last_error = caget(PREFIX + 'UVD_LAST_ERROR')
                if last_error:
                    print(f"Error: {last_error}")
                break

            # Print periodic status
            if time.time() - last_status_time >= 10.0:
                elapsed = time.time() - start_time
                emitting = caget(PREFIX + 'UVD_EMITTING')
                diode_temp = caget(PREFIX + 'UVD_DIODE_TEMP')
                print(f"[{elapsed:6.1f}s] Emitting: {'YES' if emitting else 'NO'}  "
                      f"Diode temp: {diode_temp:.2f}°C")
                last_status_time = time.time()

            # Check duration
            if duration > 0:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    print(f"\nDuration {duration:.1f}s reached")
                    break

            time.sleep(heartbeat_interval)

        # Cleanup - turn off laser
        print("\nStopping laser...")
        self.turn_off()
        return True

    def turn_off(self):
        """Immediately turn off laser"""
        if not self.wait_for_connection():
            return False

        print("Sending stop command...")
        caput(PREFIX + 'UVD_LASER_ON', 0, wait=True)

        # Wait for off state
        for i in range(50):
            emitting = caget(PREFIX + 'UVD_EMITTING')
            state = caget(PREFIX + 'UVD_LASER_STATE')

            if not emitting and state == 0:  # Not emitting and OFF
                print("Laser OFF")
                return True

            time.sleep(0.1)

        print("WARNING: Laser may still be on")
        print("Check status with: teem_laser_control.py status")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Teem Photonics Laser Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s on 10        Turn on for 10 seconds
  %(prog)s on -1        Turn on continuously (Ctrl+C to stop)
  %(prog)s off          Turn off immediately
  %(prog)s status       Show laser status
        """
    )

    parser.add_argument(
        'command',
        choices=['on', 'off', 'status'],
        help='Command to execute'
    )

    parser.add_argument(
        'duration',
        nargs='?',
        type=float,
        default=-1,
        help='Duration in seconds (-1 for continuous)'
    )

    args = parser.parse_args()

    control = LaserControl()

    if args.command == 'on':
        if args.duration == 0:
            print("ERROR: Duration cannot be 0")
            print("Use -1 for continuous operation")
            sys.exit(1)

        success = control.turn_on(args.duration)
        sys.exit(0 if success else 1)

    elif args.command == 'off':
        success = control.turn_off()
        sys.exit(0 if success else 1)

    elif args.command == 'status':
        if not control.wait_for_connection():
            sys.exit(1)
        control.print_status()
        sys.exit(0)


if __name__ == '__main__':
    main()
