#!/usr/bin/env python3
"""
Teem Photonics Laser Controller Service
Author: Claude Sonnet 4.5
Date: 2026-01-13

This service interfaces with the Teem Photonics MLC-03A-DR1 laser controller
via RS232 serial and exposes status/control via EPICS soft IOC channels.

Implements deadman switch safety mechanism requiring continuous heartbeat.
"""

import argparse
import logging
import serial
import time
import sys
from enum import Enum
from typing import Optional, Dict, Tuple
from threading import Lock

try:
    from epics import caget, caput, PV
except ImportError:
    print("ERROR: pyepics not installed. Install with: pip install pyepics")
    sys.exit(1)


# ==============================================================================
# Constants
# ==============================================================================

PREFIX = "Y1:AUX-"

# Serial communication settings (from manual page 27)
SERIAL_BAUDRATE = 19200
SERIAL_BYTESIZE = 8
SERIAL_PARITY = 'N'
SERIAL_STOPBITS = 1
SERIAL_TIMEOUT = 1.0

# Command timing (from manual - 1s between commands)
COMMAND_DELAY = 1.0

# Critical errors that trigger immediate shutdown (from manual section 5.5)
CRITICAL_ERRORS = {
    'E1', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',  # EREG1
    'E11', 'E13', 'E14', 'E15', 'E16',          # EREG2
    'E17', 'E18', 'E24'                         # EREG3
}


# ==============================================================================
# Laser State Machine
# ==============================================================================

class LaserState(Enum):
    """Laser states"""
    OFF = 0
    STARTING = 1
    ON = 2
    STOPPING = 3
    ERROR = 4


# ==============================================================================
# Teem Controller - Serial Communication
# ==============================================================================

class TeemController:
    """Handles serial communication with Teem Photonics laser controller"""

    def __init__(self, device: str, logger: logging.Logger):
        """
        Initialize serial connection

        Args:
            device: Serial device path (e.g., /dev/ttyUSB0)
            logger: Logger instance
        """
        self.device = device
        self.logger = logger
        self.serial: Optional[serial.Serial] = None
        self.lock = Lock()
        self.last_command_time = 0

        self._connect()

    def _connect(self):
        """Establish serial connection"""
        try:
            self.serial = serial.Serial(
                port=self.device,
                baudrate=SERIAL_BAUDRATE,
                bytesize=SERIAL_BYTESIZE,
                parity=SERIAL_PARITY,
                stopbits=SERIAL_STOPBITS,
                timeout=SERIAL_TIMEOUT
            )
            self.logger.info(f"Connected to {self.device}")

            # Flush buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to {self.device}: {e}")
            raise

    def _enforce_command_timing(self):
        """Ensure 1s delay between commands per manual"""
        elapsed = time.time() - self.last_command_time
        if elapsed < COMMAND_DELAY:
            time.sleep(COMMAND_DELAY - elapsed)
        self.last_command_time = time.time()

    def send_command(self, cmd_key: str, cmd: str, data: str = "") -> Optional[str]:
        """
        Send command to controller and wait for response

        Args:
            cmd_key: 'G' for Get, 'S' for Set
            cmd: 3-character command code
            data: Optional data string

        Returns:
            Response string without prompt, or None on error
        """
        with self.lock:
            try:
                self._enforce_command_timing()

                # Build command frame: [CMDKey][CMD][DATA][LF][CR]
                if data:
                    command = f"{cmd_key}{cmd}_{data}\n\r"
                else:
                    command = f"{cmd_key}{cmd}\n\r"

                self.logger.debug(f"TX: {command.strip()}")

                # Send command
                self.serial.write(command.encode('ascii'))

                # Read response until prompt '>'
                response = ""
                start_time = time.time()

                while time.time() - start_time < SERIAL_TIMEOUT:
                    char = self.serial.read(1).decode('ascii', errors='ignore')
                    if not char:
                        break
                    response += char
                    if char == '>':
                        break

                self.logger.debug(f"RX: {response.strip()}")

                # Remove prompt and whitespace
                response = response.rstrip('>\n\r ')

                return response if response else None

            except Exception as e:
                self.logger.error(f"Command failed: {e}")
                return None

    def get_status_registers(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Get status and error registers (SER command)

        Returns:
            Tuple of (EREG1, EREG2, EREG3, IREG1, IREG2, IREG3) or None
        """
        response = self.send_command('G', 'SER')

        if not response:
            return None

        try:
            # Response format: GSER_hh_hh_hh_hh_hh_hh
            parts = response.split('_')
            if len(parts) < 7:
                self.logger.warning(f"Invalid SER response: {response}")
                return None

            # Parse hex values
            ereg1 = int(parts[1], 16)
            ereg2 = int(parts[2], 16)
            ereg3 = int(parts[3], 16)
            ireg1 = int(parts[4], 16)
            ireg2 = int(parts[5], 16)
            ireg3 = int(parts[6], 16)

            return (ereg1, ereg2, ereg3, ireg1, ireg2, ireg3)

        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse SER response: {e}")
            return None

    def get_temperatures(self) -> Optional[Dict[str, float]]:
        """
        Get temperature measurements (MTE command)

        Returns:
            Dict with keys: diode_temp, crystal_temp, heatsink_temp, laser_heatsink_temp
        """
        response = self.send_command('G', 'MTE')

        if not response:
            return None

        try:
            # Response format: GMTE_dddd_dddd_dd_dd
            parts = response.split('_')
            if len(parts) < 5:
                self.logger.warning(f"Invalid MTE response: {response}")
                return None

            return {
                'diode_temp': int(parts[1]) * 0.01,      # 0.01°C resolution
                'crystal_temp': int(parts[2]) * 0.01,    # 0.01°C resolution
                'heatsink_temp': int(parts[3]),          # 1°C resolution
                'laser_heatsink_temp': int(parts[4])     # 1°C resolution
            }

        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse MTE response: {e}")
            return None

    def get_emission_time(self) -> Optional[Dict[str, int]]:
        """
        Get diode and emission runtime (EMT command)

        Returns:
            Dict with keys: diode_hours, diode_minutes, emission_hours, emission_minutes
        """
        response = self.send_command('G', 'EMT')

        if not response:
            return None

        try:
            # Response format: GEMT_ddddd_dd_ddddd_dd
            parts = response.split('_')
            if len(parts) < 5:
                self.logger.warning(f"Invalid EMT response: {response}")
                return None

            return {
                'diode_hours': int(parts[1]),
                'diode_minutes': int(parts[2]),
                'emission_hours': int(parts[3]),
                'emission_minutes': int(parts[4])
            }

        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse EMT response: {e}")
            return None

    def get_serial_number(self) -> Optional[str]:
        """Get laser serial number (SEN command)"""
        response = self.send_command('G', 'SEN')

        if not response:
            return None

        try:
            # Response format: GSEN_dddddddddddddddd
            parts = response.split('_')
            if len(parts) < 2:
                return None
            return parts[1]

        except IndexError:
            return None

    def get_firmware_versions(self) -> Optional[Tuple[str, str]]:
        """Get firmware versions (FVE command)"""
        response = self.send_command('G', 'FVE')

        if not response:
            return None

        try:
            # Response format: GFVE_ddd_ddd
            parts = response.split('_')
            if len(parts) < 3:
                return None
            return (parts[1], parts[2])  # (head_fw, controller_fw)

        except IndexError:
            return None

    def start_laser(self) -> bool:
        """
        Start laser emission (SSSD_1 command)

        Returns:
            True if command sent successfully
        """
        response = self.send_command('S', 'SSD', '1')
        success = response is not None and 'SSSD_1' in response

        if success:
            self.logger.info("Laser start command sent")
        else:
            self.logger.error("Failed to send laser start command")

        return success

    def stop_laser(self) -> bool:
        """
        Stop laser emission (SSSD_0 command)

        Returns:
            True if command sent successfully
        """
        response = self.send_command('S', 'SSD', '0')
        success = response is not None and 'SSSD_0' in response

        if success:
            self.logger.info("Laser stop command sent")
        else:
            self.logger.error("Failed to send laser stop command")

        return success

    def close(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.logger.info("Serial connection closed")


# ==============================================================================
# Error Monitor
# ==============================================================================

class ErrorMonitor:
    """Monitors error registers and triggers emergency shutdown"""

    # Error bit definitions (from manual section 5.5)
    ERROR_BITS = {
        'EREG1': {
            0: ('E1', 'UVD_ERR_HEATSINK', True),
            1: ('E2', 'UVD_ERR_LOW_VOLTAGE', False),
            2: ('E3', 'UVD_ERR_INTERLOCK', True),
            3: ('E4', 'UVD_ERR_HEAD_OVERTEMP', True),
            4: ('E5', 'UVD_ERR_DIODE_UNDERTEMP', True),
            5: ('E6', 'UVD_ERR_DIODE_OVERTEMP', True),
            6: ('E7', 'UVD_ERR_CRYSTAL_UNDERTEMP', True),
            7: ('E8', 'UVD_ERR_CRYSTAL_OVERTEMP', True),
        },
        'EREG2': {
            0: ('E9', 'UVD_ERR_TEC_OVERLOAD', False),
            1: ('E10', 'UVD_ERR_HEAD_READ', False),
            2: ('E11', 'UVD_ERR_DIODE_BOUNDARY', True),
            3: ('E12', 'UVD_ERR_HIGH_VOLTAGE', False),
            4: ('E13', 'UVD_ERR_TEC_DIODE_OPEN', True),
            5: ('E14', 'UVD_ERR_TEC_DIODE_SHORT', True),
            6: ('E15', 'UVD_ERR_TEC_CRYSTAL_OPEN', True),
            7: ('E16', 'UVD_ERR_TEC_CRYSTAL_SHORT', True),
        },
        'EREG3': {
            0: ('E17', 'UVD_ERR_DIODE_OPEN', True),
            1: ('E18', 'UVD_ERR_DIODE_SHORT', True),
            2: ('E19', 'UVD_ERR_LAMP_FAILURE', False),
            3: ('E20', 'UVD_ERR_HEAD_ID', False),
            4: ('E21', 'UVD_ERR_CROSSED_CABLES', False),
            5: ('E22', 'UVD_ERR_CONFIG', False),
            6: ('E23', 'UVD_ERR_COMM', False),
            7: ('E24', 'UVD_ERR_CRYSTAL_BOUNDARY', True),
        }
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.pv_cache = {}

    def _get_pv(self, name: str) -> PV:
        """Get cached PV object"""
        if name not in self.pv_cache:
            self.pv_cache[name] = PV(PREFIX + name)
        return self.pv_cache[name]

    def update_error_registers(self, ereg1: int, ereg2: int, ereg3: int):
        """
        Update error register PVs and individual error bits

        Args:
            ereg1, ereg2, ereg3: Error register values
        """
        # Update register PVs
        caput(PREFIX + 'UVD_EREG1', ereg1, wait=False)
        caput(PREFIX + 'UVD_EREG2', ereg2, wait=False)
        caput(PREFIX + 'UVD_EREG3', ereg3, wait=False)

        # Update individual error bits
        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG1'].items():
            value = (ereg1 >> bit_pos) & 1
            caput(PREFIX + pv_name, value, wait=False)

        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG2'].items():
            value = (ereg2 >> bit_pos) & 1
            caput(PREFIX + pv_name, value, wait=False)

        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG3'].items():
            value = (ereg3 >> bit_pos) & 1
            caput(PREFIX + pv_name, value, wait=False)

    def has_critical_error(self, ereg1: int, ereg2: int, ereg3: int) -> Tuple[bool, list]:
        """
        Check if any critical errors are present

        Returns:
            Tuple of (has_error, list_of_error_codes)
        """
        errors = []

        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG1'].items():
            if is_critical and ((ereg1 >> bit_pos) & 1):
                errors.append(error_code)

        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG2'].items():
            if is_critical and ((ereg2 >> bit_pos) & 1):
                errors.append(error_code)

        for bit_pos, (error_code, pv_name, is_critical) in self.ERROR_BITS['EREG3'].items():
            if is_critical and ((ereg3 >> bit_pos) & 1):
                errors.append(error_code)

        return (len(errors) > 0, errors)


# ==============================================================================
# Deadman Switch
# ==============================================================================

class DeadmanSwitch:
    """Implements deadman switch safety mechanism"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.heartbeat_count = 0

    def check_and_reset(self) -> bool:
        """
        Check deadman switch and auto-reset

        Returns:
            True if timeout exceeded (should stop laser)
        """
        # Get current state
        turn_off = caget(PREFIX + 'UVD_TURN_OFF')
        last_heartbeat = caget(PREFIX + 'UVD_LAST_HEARTBEAT')
        timeout = caget(PREFIX + 'UVD_HEARTBEAT_TIMEOUT')

        # If user wrote False (0), update heartbeat timestamp
        if turn_off == 0:
            caput(PREFIX + 'UVD_LAST_HEARTBEAT', time.time(), wait=False)
            self.heartbeat_count += 1
            caput(PREFIX + 'UVD_HEARTBEAT_COUNT', self.heartbeat_count, wait=False)

        # Check if timeout exceeded
        if last_heartbeat is not None:
            elapsed = time.time() - last_heartbeat
            timeout_exceeded = elapsed > timeout

            if timeout_exceeded:
                self.logger.warning(f"Heartbeat timeout exceeded: {elapsed:.1f}s > {timeout}s")
        else:
            timeout_exceeded = False

        # Auto-reset to True (safe state)
        caput(PREFIX + 'UVD_TURN_OFF', 1, wait=False)

        return timeout_exceeded


# ==============================================================================
# Laser State Machine
# ==============================================================================

class LaserStateMachine:
    """Manages laser state transitions"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.state = LaserState.OFF
        self.state_entry_time = time.time()

    def set_state(self, new_state: LaserState):
        """Transition to new state"""
        if new_state != self.state:
            self.logger.info(f"State transition: {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_entry_time = time.time()
            caput(PREFIX + 'UVD_LASER_STATE', new_state.value, wait=False)

    def get_state(self) -> LaserState:
        """Get current state"""
        return self.state

    def time_in_state(self) -> float:
        """Get time spent in current state (seconds)"""
        return time.time() - self.state_entry_time


# ==============================================================================
# Main Service
# ==============================================================================

class TeemLaserService:
    """Main service orchestrator"""

    def __init__(self, device: str, log_level: str):
        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Initialize components
        self.logger.info("Initializing Teem Laser Service")
        self.controller = TeemController(device, self.logger)
        self.error_monitor = ErrorMonitor(self.logger)
        self.deadman = DeadmanSwitch(self.logger)
        self.state_machine = LaserStateMachine(self.logger)

        # Service state
        self.running = True
        self.start_time = time.time()

        # Initialize PVs
        self._initialize_pvs()

    def _setup_logging(self, level_str: str) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('TeemLaser')

        level = getattr(logging, level_str.upper(), logging.INFO)
        logger.setLevel(level)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        try:
            file_handler = logging.FileHandler('/home/controls/labutils/teem_laser.log')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")

        return logger

    def _initialize_pvs(self):
        """Initialize PVs with startup values"""
        # Get system info
        serial_num = self.controller.get_serial_number()
        if serial_num:
            caput(PREFIX + 'UVD_SERIAL_NUMBER', serial_num, wait=False)
            self.logger.info(f"Serial number: {serial_num}")

        firmware = self.controller.get_firmware_versions()
        if firmware:
            caput(PREFIX + 'UVD_FW_HEAD', f"V{firmware[0]}", wait=False)
            caput(PREFIX + 'UVD_FW_CONTROLLER', f"V{firmware[1]}", wait=False)
            self.logger.info(f"Firmware: Head={firmware[0]}, Controller={firmware[1]}")

        # Initialize heartbeat
        caput(PREFIX + 'UVD_LAST_HEARTBEAT', time.time(), wait=False)

    def poll_controller(self):
        """Poll controller for status updates"""
        # Get status registers
        status = self.controller.get_status_registers()
        if status:
            ereg1, ereg2, ereg3, ireg1, ireg2, ireg3 = status

            # Update error registers and check for critical errors
            self.error_monitor.update_error_registers(ereg1, ereg2, ereg3)
            has_error, error_codes = self.error_monitor.has_critical_error(ereg1, ereg2, ereg3)

            if has_error:
                self.logger.error(f"Critical errors detected: {error_codes}")
                caput(PREFIX + 'UVD_LAST_ERROR', f"Critical: {','.join(error_codes)}", wait=False)
                self.state_machine.set_state(LaserState.ERROR)
                self.controller.stop_laser()

            # Update info registers
            caput(PREFIX + 'UVD_IREG1', ireg1, wait=False)
            caput(PREFIX + 'UVD_IREG2', ireg2, wait=False)
            caput(PREFIX + 'UVD_IREG3', ireg3, wait=False)

            # Update status bits (from info registers)
            # IREG2:I11 = Ready for emission
            ready = (ireg2 >> 2) & 1
            caput(PREFIX + 'UVD_READY', ready, wait=False)
            caput(PREFIX + 'UVD_TEMP_OK', ready, wait=False)

            # IREG2:I9 = Laser diode current ON
            emitting = ireg2 & 1
            caput(PREFIX + 'UVD_EMITTING', emitting, wait=False)

        # Get temperatures
        temps = self.controller.get_temperatures()
        if temps:
            caput(PREFIX + 'UVD_DIODE_TEMP', temps['diode_temp'], wait=False)
            caput(PREFIX + 'UVD_CRYSTAL_TEMP', temps['crystal_temp'], wait=False)
            caput(PREFIX + 'UVD_HEATSINK_TEMP', temps['heatsink_temp'], wait=False)
            caput(PREFIX + 'UVD_LASER_HEATSINK_TEMP', temps['laser_heatsink_temp'], wait=False)

        # Get emission time (less frequently)
        if int(time.time()) % 10 == 0:  # Every 10 seconds
            emission = self.controller.get_emission_time()
            if emission:
                caput(PREFIX + 'UVD_DIODE_HOURS', emission['diode_hours'], wait=False)
                caput(PREFIX + 'UVD_DIODE_MINUTES', emission['diode_minutes'], wait=False)
                caput(PREFIX + 'UVD_EMISSION_HOURS', emission['emission_hours'], wait=False)
                caput(PREFIX + 'UVD_EMISSION_MINUTES', emission['emission_minutes'], wait=False)

    def process_commands(self):
        """Process commands from IOC"""
        laser_on_cmd = caget(PREFIX + 'UVD_LASER_ON')
        emergency_stop = caget(PREFIX + 'UVD_EMERGENCY_STOP')

        current_state = self.state_machine.get_state()

        # Emergency stop overrides everything
        if emergency_stop == 1:
            self.logger.warning("EMERGENCY STOP triggered")
            self.controller.stop_laser()
            self.state_machine.set_state(LaserState.ERROR)
            caput(PREFIX + 'UVD_EMERGENCY_STOP', 0, wait=False)
            caput(PREFIX + 'UVD_LASER_ON', 0, wait=False)
            return

        # State machine logic
        if current_state == LaserState.OFF:
            if laser_on_cmd == 1:
                self.logger.info("Laser ON command received")
                if self.controller.start_laser():
                    self.state_machine.set_state(LaserState.STARTING)

        elif current_state == LaserState.STARTING:
            # Wait for 5-second startup delay (per manual)
            if self.state_machine.time_in_state() >= 5.0:
                ready = caget(PREFIX + 'UVD_READY')
                if ready == 1:
                    self.state_machine.set_state(LaserState.ON)
                    self.logger.info("Laser emission started")
                else:
                    self.logger.warning("Laser not ready after 5s startup delay")
                    self.state_machine.set_state(LaserState.ERROR)

        elif current_state == LaserState.ON:
            if laser_on_cmd == 0:
                self.logger.info("Laser OFF command received")
                self.controller.stop_laser()
                self.state_machine.set_state(LaserState.STOPPING)

        elif current_state == LaserState.STOPPING:
            emitting = caget(PREFIX + 'UVD_EMITTING')
            if emitting == 0:
                self.state_machine.set_state(LaserState.OFF)
                self.logger.info("Laser emission stopped")

        elif current_state == LaserState.ERROR:
            # Must manually reset from error state
            if laser_on_cmd == 0:
                self.state_machine.set_state(LaserState.OFF)
                caput(PREFIX + 'UVD_LAST_ERROR', "", wait=False)

    def check_deadman(self):
        """Check deadman switch"""
        current_state = self.state_machine.get_state()

        # Only active when laser is ON
        if current_state == LaserState.ON:
            timeout_exceeded = self.deadman.check_and_reset()

            if timeout_exceeded:
                self.logger.warning("Deadman switch timeout - stopping laser")
                caput(PREFIX + 'UVD_LASER_ON', 0, wait=False)
                caput(PREFIX + 'UVD_LAST_ERROR', "Deadman timeout", wait=False)
                self.controller.stop_laser()
                self.state_machine.set_state(LaserState.STOPPING)
        else:
            # Not ON - just reset the channel
            self.deadman.check_and_reset()

    def update_diagnostics(self):
        """Update diagnostic PVs"""
        uptime = time.time() - self.start_time
        caput(PREFIX + 'UVD_UPTIME', uptime, wait=False)

    def run(self):
        """Main service loop"""
        self.logger.info("Teem Laser Service started")

        loop_count = 0

        try:
            while self.running:
                # Main loop runs at 10 Hz for responsive deadman switch
                loop_start = time.time()

                # Poll controller every second
                if loop_count % 10 == 0:
                    self.poll_controller()

                # Process commands every loop
                self.process_commands()

                # Check deadman every loop
                self.check_deadman()

                # Update diagnostics every 10 seconds
                if loop_count % 100 == 0:
                    self.update_diagnostics()

                loop_count += 1

                # Sleep to maintain 10 Hz
                elapsed = time.time() - loop_start
                sleep_time = max(0, 0.1 - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down service")

        # Stop laser if running
        if self.state_machine.get_state() in [LaserState.ON, LaserState.STARTING]:
            self.logger.info("Stopping laser before shutdown")
            self.controller.stop_laser()
            time.sleep(1)

        # Close serial connection
        self.controller.close()

        self.logger.info("Service stopped")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Teem Photonics Laser Controller Service'
    )
    parser.add_argument(
        '--device',
        default='/dev/ttyUSB0',
        help='Serial device path (default: /dev/ttyUSB0)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Create and run service
    service = TeemLaserService(args.device, args.log_level)
    service.run()


if __name__ == '__main__':
    main()
